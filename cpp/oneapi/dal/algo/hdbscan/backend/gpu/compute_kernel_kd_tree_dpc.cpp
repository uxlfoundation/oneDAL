/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// Memory-efficient GPU HDBSCAN using blocked distance computation.
/// Avoids the O(N²) distance matrix of the brute_force method by:
///   1. Computing core distances in blocks of B rows (B×N GEMM + kselect)
///   2. Running Prim's MST on host with on-the-fly MRD computation
///   3. Reusing existing sort_mst_by_weight and extract_clusters GPU kernels

#include "oneapi/dal/algo/hdbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/results.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

/// Choose block size so that the B×N distance block fits in ~256 MB of device memory.
/// For Float=float (4 bytes): block_size = 256MB / (N * 4) = 64M / N.
/// Clamped to [256, N] range.
static std::int64_t choose_block_size(std::int64_t row_count, std::int64_t float_size) {
    const std::int64_t target_bytes = 256 * 1024 * 1024; // 256 MB
    std::int64_t bs = target_bytes / (row_count * float_size);
    if (bs < 256)
        bs = 256;
    if (bs > row_count)
        bs = row_count;
    return bs;
}

template <typename Float>
static result_t compute_kernel_kd_tree_impl(const context_gpu& ctx,
                                            const descriptor_t& desc,
                                            const table& local_data) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_kd_tree, ctx.get_queue());

    auto& queue = ctx.get_queue();

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t col_count = local_data.get_column_count();
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int64_t min_samples = desc.get_min_samples();
    const std::int64_t edge_count = row_count - 1;

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    // =========================================================================
    // Step 1: Blocked core distance computation
    // Process block_size rows at a time: compute B×N squared distances via GEMM,
    // run kselect to find k-th nearest per row, take sqrt.
    // Peak memory: O(B×N) + O(B×k) instead of O(N²).
    // =========================================================================

    const std::int64_t block_size = choose_block_size(row_count, sizeof(Float));
    const std::int64_t k = min_samples; // includes self-distance at 0

    auto [core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    sycl::event ev_core_dist = core_dist_event;

    // Compute norms once (reused for each block's GEMM distance)
    auto [norms, norms_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    sycl::event ev_norms = norms_event;

    // Compute squared norms: norms[i] = sum(data[i,:]^2)
    const Float* data_ptr = data_nd.get_data();
    Float* norms_ptr = norms.get_mutable_data();
    const std::int64_t d = col_count;

    auto norms_compute_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ev_norms });
        h.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            Float s = Float(0);
            for (std::int64_t f = 0; f < d; f++) {
                const Float v = data_ptr[i * d + f];
                s += v * v;
            }
            norms_ptr[i] = s;
        });
    });

    Float* core_ptr = core_distances.get_mutable_data();
    sycl::event prev_block_event = ev_core_dist;

    for (std::int64_t b_start = 0; b_start < row_count; b_start += block_size) {
        const std::int64_t b_end = std::min(b_start + block_size, row_count);
        const std::int64_t b_rows = b_end - b_start;

        // Create a view of the block rows: data_nd[b_start:b_end, :]
        auto block_view = pr::ndview<Float, 2>::wrap(data_nd.get_data() + b_start * col_count,
                                                     { b_rows, col_count });

        // Allocate B×N squared distance block
        auto [sq_dist_block, sq_dist_block_event] =
            pr::ndarray<Float, 2>::zeros(queue, { b_rows, row_count }, sycl::usm::alloc::device);
        sycl::event ev_sq_dist_block = sq_dist_block_event;

        // Compute squared distances: sq_dist_block[i,j] = ||block[i] - data[j]||²
        // Using the norms: sq_dist[i,j] = norms_block[i] + norms[j] - 2 * block[i] · data[j]^T
        pr::squared_l2_distance<Float> dist_op(queue);
        auto dist_event = dist_op(block_view,
                                  data_nd,
                                  sq_dist_block,
                                  { ev_sq_dist_block, norms_compute_event, prev_block_event });

        // Kselect: find k smallest squared distances per row
        auto [ksel_vals, ksel_vals_event] =
            pr::ndarray<Float, 2>::zeros(queue, { b_rows, k }, sycl::usm::alloc::device);
        sycl::event ev_ksel_vals = ksel_vals_event;

        pr::kselect_by_rows<Float> ksel(queue, { b_rows, row_count }, k);
        auto ksel_event = ksel(queue, sq_dist_block, k, ksel_vals, { dist_event, ev_ksel_vals });

        // Extract k-th value (column k-1) per row, sqrt, store as core distance
        const Float* ksel_ptr = ksel_vals.get_data();
        const std::int64_t kk = k;
        const std::int64_t offset = b_start;

        auto extract_event = queue.submit([&](sycl::handler& h) {
            h.depends_on({ ksel_event });
            h.parallel_for(sycl::range<1>(b_rows), [=](sycl::id<1> idx) {
                const std::int64_t i = idx[0];
                const Float sq_val = ksel_ptr[i * kk + (kk - 1)];
                core_ptr[offset + i] = sycl::sqrt(sycl::fmax(sq_val, Float(0)));
            });
        });

        prev_block_event = extract_event;
    }

    // Wait for all core distances to be computed
    prev_block_event.wait_and_throw();

    // =========================================================================
    // Step 2: Host-side Prim's MST with on-the-fly MRD computation
    // Copy data + core distances to host. For each Prim's iteration, compute
    // MRD(best, j) = max(core[best], core[j], euclidean(data[best], data[j]))
    // on-the-fly. Memory: O(N*D + N) instead of O(N²).
    // =========================================================================

    auto data_host = data_nd.to_host(queue);
    auto core_host = core_distances.to_host(queue);

    const Float* h_data = data_host.get_data();
    const Float* h_core = core_host.get_data();
    const std::int64_t n = row_count;

    std::vector<Float> min_edge_vec(n, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from_vec(n, 0);
    std::vector<char> in_mst_vec(n, 0);

    std::vector<std::int32_t> from_vec(edge_count);
    std::vector<std::int32_t> to_vec(edge_count);
    std::vector<Float> weight_vec(edge_count);

    // Helper: compute Euclidean distance between points i and j
    auto eucl_dist = [&](std::int64_t i, std::int64_t j) -> Float {
        Float s = Float(0);
        for (std::int64_t f = 0; f < d; f++) {
            const Float diff = h_data[i * d + f] - h_data[j * d + f];
            s += diff * diff;
        }
        return std::sqrt(s);
    };

    // Helper: compute MRD between points i and j
    auto mrd = [&](std::int64_t i, std::int64_t j) -> Float {
        const Float ed = eucl_dist(i, j);
        return std::max({ h_core[i], h_core[j], ed });
    };

    // Initialize: start from node 0
    in_mst_vec[0] = 1;
    for (std::int64_t j = 1; j < n; j++) {
        min_edge_vec[j] = mrd(0, j);
    }

    for (std::int64_t e = 0; e < edge_count; e++) {
        // Find minimum-weight edge to a non-MST node
        std::int32_t best = -1;
        Float best_w = std::numeric_limits<Float>::max();
        for (std::int64_t j = 0; j < n; j++) {
            if (!in_mst_vec[j] && min_edge_vec[j] < best_w) {
                best_w = min_edge_vec[j];
                best = static_cast<std::int32_t>(j);
            }
        }

        from_vec[e] = min_from_vec[best];
        to_vec[e] = best;
        weight_vec[e] = best_w;
        in_mst_vec[best] = 1;

        // Update edges: check if new node 'best' provides shorter path to remaining nodes
        for (std::int64_t j = 0; j < n; j++) {
            if (in_mst_vec[j])
                continue;
            const Float mrd_val = mrd(best, j);
            if (mrd_val < min_edge_vec[j]) {
                min_edge_vec[j] = mrd_val;
                min_from_vec[j] = best;
            }
        }
    }

    // =========================================================================
    // Step 3: Copy MST to device + reuse existing GPU kernels
    // =========================================================================

    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    // Wait for allocations before memcpy
    sycl::event::wait({ mst_from_event, mst_to_event, mst_weights_event });

    auto from_copy = queue.memcpy(mst_from.get_mutable_data(),
                                  from_vec.data(),
                                  edge_count * sizeof(std::int32_t));
    auto to_copy =
        queue.memcpy(mst_to.get_mutable_data(), to_vec.data(), edge_count * sizeof(std::int32_t));
    auto weight_copy =
        queue.memcpy(mst_weights.get_mutable_data(), weight_vec.data(), edge_count * sizeof(Float));

    sycl::event::wait({ from_copy, to_copy, weight_copy });

    // Step 4: Sort MST edges by weight (reuses existing GPU kernel)
    auto sort_event =
        kernels_fp<Float>::sort_mst_by_weight(queue, mst_from, mst_to, mst_weights, edge_count, {});

    // Step 5: Extract flat clusters using EOM (reuses existing GPU kernel)
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    sycl::event ev_responses = responses_event;

    auto cluster_event = kernels_fp<Float>::extract_clusters(queue,
                                                             mst_from,
                                                             mst_to,
                                                             mst_weights,
                                                             arr_responses,
                                                             row_count,
                                                             min_cluster_size,
                                                             { sort_event, ev_responses });
    cluster_event.wait_and_throw();

    // Count clusters from responses on device
    auto responses_host = arr_responses.to_host(queue);
    const auto* resp_ptr = responses_host.get_data();

    std::int32_t max_label = -1;
    for (std::int64_t i = 0; i < row_count; i++) {
        if (resp_ptr[i] > max_label)
            max_label = resp_ptr[i];
    }
    const std::int64_t cluster_count = (max_label >= 0) ? (max_label + 1) : 0;

    return make_results<Float>(queue, desc, arr_responses, cluster_count);
}

template <typename Float>
struct compute_kernel_gpu<Float, method::kd_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_kd_tree_impl<Float>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_gpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::kd_tree, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
