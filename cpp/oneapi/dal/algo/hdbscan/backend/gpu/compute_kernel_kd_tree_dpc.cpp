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

/// Memory-efficient GPU HDBSCAN using blocked distance computation + Boruvka MST.
/// Avoids the O(N²) distance matrix of the brute_force method by:
///   1. Computing core distances in blocks of B rows (B×N GEMM + kselect)
///   2. Building kd-tree on host, running Boruvka MST with tree-accelerated queries
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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace oneapi::dal::hdbscan::backend {

// =========================================================================
// Host-side kd-tree for Boruvka MST (used after GPU computes core distances)
// =========================================================================

template <typename FP>
struct HostKdNode {
    std::int32_t split_dim; // -1 for leaf
    FP split_val;
    std::int32_t left, right;
    std::int32_t point_begin, point_end;
    std::int32_t component_id; // -1 = mixed, >= 0 = uniform
};

template <typename FP>
static std::int32_t host_build_kd_tree(const FP* data,
                                       std::int32_t* indices,
                                       std::int32_t begin,
                                       std::int32_t end,
                                       std::int32_t n_cols,
                                       std::vector<HostKdNode<FP>>& nodes,
                                       FP* bbox_lo,
                                       FP* bbox_hi,
                                       std::int32_t max_leaf) {
    const auto idx = static_cast<std::int32_t>(nodes.size());
    nodes.push_back({});
    auto& node = nodes[idx];
    node.point_begin = begin;
    node.point_end = end;
    node.component_id = -1;

    FP best_spread = FP(-1);
    std::int32_t best_dim = 0;
    for (std::int32_t d = 0; d < n_cols; d++) {
        FP lo = std::numeric_limits<FP>::max();
        FP hi = std::numeric_limits<FP>::lowest();
        for (std::int32_t i = begin; i < end; i++) {
            const FP val = data[indices[i] * n_cols + d];
            if (val < lo)
                lo = val;
            if (val > hi)
                hi = val;
        }
        bbox_lo[idx * n_cols + d] = lo;
        bbox_hi[idx * n_cols + d] = hi;
        const FP spread = hi - lo;
        if (spread > best_spread) {
            best_spread = spread;
            best_dim = d;
        }
    }

    const std::int32_t count = end - begin;
    if (count <= max_leaf) {
        node.split_dim = -1;
        node.split_val = FP(0);
        node.left = -1;
        node.right = -1;
        return idx;
    }

    node.split_dim = best_dim;
    const std::int32_t mid = begin + count / 2;
    std::nth_element(indices + begin,
                     indices + mid,
                     indices + end,
                     [&](std::int32_t a, std::int32_t b) {
                         return data[a * n_cols + best_dim] < data[b * n_cols + best_dim];
                     });
    node.split_val = data[indices[mid] * n_cols + best_dim];
    node.left =
        host_build_kd_tree(data, indices, begin, mid, n_cols, nodes, bbox_lo, bbox_hi, max_leaf);
    node.right =
        host_build_kd_tree(data, indices, mid, end, n_cols, nodes, bbox_lo, bbox_hi, max_leaf);
    return idx;
}

template <typename FP>
static FP host_compute_min_core_dists(const std::vector<HostKdNode<FP>>& nodes,
                                      const std::int32_t* indices,
                                      const FP* core_dists,
                                      std::vector<FP>& min_core,
                                      std::int32_t idx) {
    const auto& node = nodes[idx];
    if (node.split_dim < 0) {
        FP mn = std::numeric_limits<FP>::max();
        for (std::int32_t i = node.point_begin; i < node.point_end; i++)
            mn = std::min(mn, core_dists[indices[i]]);
        min_core[idx] = mn;
        return mn;
    }
    FP l = host_compute_min_core_dists(nodes, indices, core_dists, min_core, node.left);
    FP r = host_compute_min_core_dists(nodes, indices, core_dists, min_core, node.right);
    min_core[idx] = std::min(l, r);
    return min_core[idx];
}

template <typename FP>
static std::int32_t host_update_components(std::vector<HostKdNode<FP>>& nodes,
                                           const std::int32_t* indices,
                                           const std::int32_t* comp_of,
                                           std::int32_t idx) {
    auto& node = nodes[idx];
    if (node.split_dim < 0) {
        const std::int32_t first = comp_of[indices[node.point_begin]];
        for (std::int32_t i = node.point_begin + 1; i < node.point_end; i++) {
            if (comp_of[indices[i]] != first) {
                node.component_id = -1;
                return -1;
            }
        }
        node.component_id = first;
        return first;
    }
    std::int32_t lc = host_update_components(nodes, indices, comp_of, node.left);
    std::int32_t rc = host_update_components(nodes, indices, comp_of, node.right);
    if (lc >= 0 && lc == rc) {
        node.component_id = lc;
        return lc;
    }
    node.component_id = -1;
    return -1;
}

template <typename FP>
static void host_nearest_mrd_query(const FP* data,
                                   std::int32_t n_cols,
                                   const std::vector<HostKdNode<FP>>& nodes,
                                   const std::int32_t* indices,
                                   const FP* core_dists,
                                   const FP* bbox_lo,
                                   const FP* bbox_hi,
                                   const FP* min_core,
                                   const std::int32_t* comp_of,
                                   const FP* query_pt,
                                   FP query_core,
                                   std::int32_t query_comp,
                                   std::int32_t node_idx,
                                   FP& best_mrd,
                                   std::int32_t& best_idx) {
    const auto& node = nodes[node_idx];

    if (node.component_id == query_comp)
        return;

    // Bounding-box MRD lower bound
    FP min_sq = FP(0);
    const FP* lo = bbox_lo + node_idx * n_cols;
    const FP* hi = bbox_hi + node_idx * n_cols;
    for (std::int32_t dd = 0; dd < n_cols; dd++) {
        FP excess = FP(0);
        if (query_pt[dd] < lo[dd])
            excess = lo[dd] - query_pt[dd];
        else if (query_pt[dd] > hi[dd])
            excess = query_pt[dd] - hi[dd];
        min_sq += excess * excess;
    }
    FP mrd_lb = std::sqrt(min_sq);
    if (query_core > mrd_lb)
        mrd_lb = query_core;
    if (min_core[node_idx] > mrd_lb)
        mrd_lb = min_core[node_idx];
    if (mrd_lb >= best_mrd)
        return;

    if (node.split_dim < 0) {
        for (std::int32_t i = node.point_begin; i < node.point_end; i++) {
            const std::int32_t pi = indices[i];
            if (comp_of[pi] == query_comp)
                continue;
            const FP* row = data + pi * n_cols;
            FP dist = FP(0);
            for (std::int32_t dd = 0; dd < n_cols; dd++) {
                const FP df = query_pt[dd] - row[dd];
                dist += df * df;
            }
            dist = std::sqrt(dist);
            FP mrd_val = dist;
            if (query_core > mrd_val)
                mrd_val = query_core;
            if (core_dists[pi] > mrd_val)
                mrd_val = core_dists[pi];
            if (mrd_val < best_mrd) {
                best_mrd = mrd_val;
                best_idx = pi;
            }
        }
        return;
    }

    const FP diff = query_pt[node.split_dim] - node.split_val;
    const std::int32_t near = (diff <= FP(0)) ? node.left : node.right;
    const std::int32_t far = (diff <= FP(0)) ? node.right : node.left;
    host_nearest_mrd_query(data,
                           n_cols,
                           nodes,
                           indices,
                           core_dists,
                           bbox_lo,
                           bbox_hi,
                           min_core,
                           comp_of,
                           query_pt,
                           query_core,
                           query_comp,
                           near,
                           best_mrd,
                           best_idx);
    host_nearest_mrd_query(data,
                           n_cols,
                           nodes,
                           indices,
                           core_dists,
                           bbox_lo,
                           bbox_hi,
                           min_core,
                           comp_of,
                           query_pt,
                           query_core,
                           query_comp,
                           far,
                           best_mrd,
                           best_idx);
}

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
    // Step 2: Host-side Boruvka MST with kd-tree accelerated queries
    // Copy data + core distances to host. Build kd-tree, run O(N log^2 N)
    // Boruvka's algorithm instead of O(N^2) Prim's.
    // =========================================================================

    auto data_host = data_nd.to_host(queue);
    auto core_host = core_distances.to_host(queue);

    const Float* h_data = data_host.get_data();
    const Float* h_core = core_host.get_data();
    const std::int64_t n = row_count;
    const std::int32_t n32 = static_cast<std::int32_t>(n);
    const std::int32_t d32 = static_cast<std::int32_t>(d);

    // Build kd-tree on host
    const std::int32_t max_leaf = 40;
    std::vector<HostKdNode<Float>> tree_nodes;
    tree_nodes.reserve(4 * n32);
    std::vector<std::int32_t> pt_indices(n32);
    std::iota(pt_indices.begin(), pt_indices.end(), 0);

    // Pre-allocate bbox arrays (will grow with tree_nodes via push_back)
    const std::int64_t max_nodes_est = 4 * n;
    std::vector<Float> bbox_lo(max_nodes_est * d, Float(0));
    std::vector<Float> bbox_hi(max_nodes_est * d, Float(0));

    host_build_kd_tree(h_data,
                       pt_indices.data(),
                       0,
                       n32,
                       d32,
                       tree_nodes,
                       bbox_lo.data(),
                       bbox_hi.data(),
                       max_leaf);

    // Compute per-node minimum core distances
    std::vector<Float> min_core_node(tree_nodes.size(), std::numeric_limits<Float>::max());
    host_compute_min_core_dists(tree_nodes, pt_indices.data(), h_core, min_core_node, 0);

    // Boruvka MST
    std::vector<std::int32_t> uf_parent(n32), uf_rank(n32, 0), comp_of(n32);
    std::iota(uf_parent.begin(), uf_parent.end(), 0);
    std::iota(comp_of.begin(), comp_of.end(), 0);

    auto uf_find = [&](std::int32_t x) -> std::int32_t {
        while (uf_parent[x] != x) {
            uf_parent[x] = uf_parent[uf_parent[x]];
            x = uf_parent[x];
        }
        return x;
    };
    auto uf_union = [&](std::int32_t rx, std::int32_t ry) {
        if (uf_rank[rx] < uf_rank[ry])
            uf_parent[rx] = ry;
        else if (uf_rank[rx] > uf_rank[ry])
            uf_parent[ry] = rx;
        else {
            uf_parent[ry] = rx;
            uf_rank[rx]++;
        }
    };

    host_update_components(tree_nodes, pt_indices.data(), comp_of.data(), 0);

    std::vector<std::int32_t> from_vec(edge_count);
    std::vector<std::int32_t> to_vec(edge_count);
    std::vector<Float> weight_vec(edge_count);

    std::vector<Float> pt_best_mrd(n32);
    std::vector<std::int32_t> pt_best_idx(n32);
    std::vector<Float> comp_best_mrd(n32);
    std::vector<std::int32_t> comp_best_from(n32), comp_best_to(n32);

    std::int64_t edges_added = 0;
    std::int64_t num_components = n;

    while (num_components > 1) {
        // Phase 1: per-point nearest different-component query
        for (std::int32_t i = 0; i < n32; i++) {
            Float bm = std::numeric_limits<Float>::max();
            std::int32_t bi = -1;
            host_nearest_mrd_query(h_data,
                                   d32,
                                   tree_nodes,
                                   pt_indices.data(),
                                   h_core,
                                   bbox_lo.data(),
                                   bbox_hi.data(),
                                   min_core_node.data(),
                                   comp_of.data(),
                                   h_data + i * d32,
                                   h_core[i],
                                   comp_of[i],
                                   0,
                                   bm,
                                   bi);
            pt_best_mrd[i] = bm;
            pt_best_idx[i] = bi;
        }

        // Phase 2: per-component best edge
        std::fill(comp_best_mrd.begin(), comp_best_mrd.end(), std::numeric_limits<Float>::max());
        std::fill(comp_best_from.begin(), comp_best_from.end(), -1);
        std::fill(comp_best_to.begin(), comp_best_to.end(), -1);
        for (std::int32_t i = 0; i < n32; i++) {
            if (pt_best_idx[i] < 0)
                continue;
            const std::int32_t c = comp_of[i];
            if (pt_best_mrd[i] < comp_best_mrd[c]) {
                comp_best_mrd[c] = pt_best_mrd[i];
                comp_best_from[c] = i;
                comp_best_to[c] = pt_best_idx[i];
            }
        }

        // Phase 3: merge
        std::int64_t added = 0;
        for (std::int32_t c = 0; c < n32; c++) {
            if (comp_best_from[c] < 0)
                continue;
            const std::int32_t u = comp_best_from[c], v = comp_best_to[c];
            const std::int32_t ru = uf_find(u), rv = uf_find(v);
            if (ru == rv)
                continue;
            from_vec[edges_added] = u;
            to_vec[edges_added] = v;
            weight_vec[edges_added] = comp_best_mrd[c];
            edges_added++;
            added++;
            uf_union(ru, rv);
            num_components--;
        }
        if (added == 0)
            break;

        // Phase 4-5: flatten labels + update tree
        for (std::int32_t i = 0; i < n32; i++)
            comp_of[i] = uf_find(i);
        host_update_components(tree_nodes, pt_indices.data(), comp_of.data(), 0);
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
