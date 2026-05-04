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

/// GPU HDBSCAN kd_tree/ball_tree: blocked core distance computation + GPU Boruvka MST.
/// No host-side tree construction or MST — all computation stays on device.

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
static std::int64_t choose_block_size(std::int64_t row_count, std::int64_t float_size) {
    const std::int64_t target_bytes = 256 * 1024 * 1024;
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
    const auto metric = desc.get_metric();
    const double degree = desc.get_degree();
    const std::int32_t cluster_selection =
        (desc.get_cluster_selection() == cluster_selection_method::leaf) ? 1 : 0;
    const bool allow_single_cluster = desc.get_allow_single_cluster();
    const double cluster_selection_epsilon = desc.get_cluster_selection_epsilon();
    const std::int64_t max_cluster_size = desc.get_max_cluster_size();
    const double alpha = desc.get_alpha();

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    // Step 1: Blocked core distance computation on GPU
    const std::int64_t block_size = choose_block_size(row_count, sizeof(Float));
    const std::int64_t k = min_samples;
    const bool needs_sqrt = (metric == distance_metric::euclidean);

    auto [core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);

    Float* core_ptr = core_distances.get_mutable_data();
    sycl::event prev_block_event = core_dist_event;

    for (std::int64_t b_start = 0; b_start < row_count; b_start += block_size) {
        const std::int64_t b_end = std::min(b_start + block_size, row_count);
        const std::int64_t b_rows = b_end - b_start;

        auto block_view = pr::ndview<Float, 2>::wrap(data_nd.get_data() + b_start * col_count,
                                                     { b_rows, col_count });

        auto [dist_block, dist_block_event] =
            pr::ndarray<Float, 2>::zeros(queue, { b_rows, row_count }, sycl::usm::alloc::device);

        sycl::event dist_event;
        switch (metric) {
            case distance_metric::manhattan: {
                pr::distance<Float, pr::lp_metric<Float>> dist_op(queue,
                                                                  pr::lp_metric<Float>(Float(1)));
                dist_event = dist_op(block_view,
                                     data_nd,
                                     dist_block,
                                     { dist_block_event, prev_block_event });
                break;
            }
            case distance_metric::minkowski: {
                pr::distance<Float, pr::lp_metric<Float>> dist_op(
                    queue,
                    pr::lp_metric<Float>(static_cast<Float>(degree)));
                dist_event = dist_op(block_view,
                                     data_nd,
                                     dist_block,
                                     { dist_block_event, prev_block_event });
                break;
            }
            case distance_metric::chebyshev: {
                pr::chebyshev_distance<Float> dist_op(queue);
                dist_event = dist_op(block_view,
                                     data_nd,
                                     dist_block,
                                     { dist_block_event, prev_block_event });
                break;
            }
            default: {
                pr::squared_l2_distance<Float> dist_op(queue);
                dist_event = dist_op(block_view,
                                     data_nd,
                                     dist_block,
                                     { dist_block_event, prev_block_event });
                break;
            }
        }

        auto [ksel_vals, ksel_vals_event] =
            pr::ndarray<Float, 2>::zeros(queue, { b_rows, k }, sycl::usm::alloc::device);

        pr::kselect_by_rows<Float> ksel(queue, { b_rows, row_count }, k);
        auto ksel_event = ksel(queue, dist_block, k, ksel_vals, { dist_event, ksel_vals_event });

        const Float* ksel_ptr = ksel_vals.get_data();
        const std::int64_t kk = k;
        const std::int64_t offset = b_start;

        auto extract_event = queue.submit([&](sycl::handler& h) {
            h.depends_on({ ksel_event });
            h.parallel_for(sycl::range<1>(b_rows), [=](sycl::id<1> idx) {
                const std::int64_t i = idx[0];
                const Float val = ksel_ptr[i * kk + (kk - 1)];
                core_ptr[offset + i] = needs_sqrt ? sycl::sqrt(sycl::fmax(val, Float(0))) : val;
            });
        });

        prev_block_event = extract_event;
    }

    // Step 2: GPU Boruvka MST with on-the-fly distance computation
    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    auto mst_event = kernels_fp<Float>::build_mst_otf(
        queue,
        data_nd,
        core_distances,
        mst_from,
        mst_to,
        mst_weights,
        row_count,
        col_count,
        metric,
        degree,
        { prev_block_event, mst_from_event, mst_to_event, mst_weights_event });

    // Step 2b: Apply alpha scaling to MST weights
    if (alpha != 1.0) {
        const Float inv_alpha = static_cast<Float>(1.0 / alpha);
        Float* w_ptr = mst_weights.get_mutable_data();
        auto alpha_event = queue.submit([&](sycl::handler& h) {
            h.depends_on({ mst_event });
            h.parallel_for(sycl::range<1>(edge_count), [=](sycl::id<1> idx) {
                w_ptr[idx[0]] *= inv_alpha;
            });
        });
        mst_event = alpha_event;
    }

    // Step 3: Sort MST edges by weight
    auto sort_event = kernels_fp<Float>::sort_mst_by_weight(queue,
                                                            mst_from,
                                                            mst_to,
                                                            mst_weights,
                                                            edge_count,
                                                            { mst_event });

    // Step 4: Extract flat clusters
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);

    auto cluster_event = kernels_fp<Float>::extract_clusters(queue,
                                                             mst_from,
                                                             mst_to,
                                                             mst_weights,
                                                             arr_responses,
                                                             row_count,
                                                             min_cluster_size,
                                                             { sort_event, responses_event },
                                                             cluster_selection,
                                                             allow_single_cluster,
                                                             cluster_selection_epsilon,
                                                             max_cluster_size);
    cluster_event.wait_and_throw();

    // Count clusters via GPU reduction
    auto [max_label_arr, ml_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, -1, sycl::usm::alloc::device);
    sycl::event ml_alloc_ev = ml_ev;
    const std::int32_t* r_ptr = arr_responses.get_data();
    std::int32_t* ml_ptr = max_label_arr.get_mutable_data();
    auto reduce_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ml_alloc_ev });
        h.single_task([=]() {
            std::int32_t mx = -1;
            for (std::int64_t i = 0; i < row_count; i++) {
                if (r_ptr[i] > mx)
                    mx = r_ptr[i];
            }
            ml_ptr[0] = mx;
        });
    });
    auto max_label_host = max_label_arr.to_host(queue, { reduce_event });
    const std::int32_t max_label = max_label_host.get_data()[0];
    const std::int64_t cluster_count = (max_label >= 0) ? (max_label + 1) : 0;

    return make_results<Float>(queue, desc, arr_responses, cluster_count, local_data);
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
