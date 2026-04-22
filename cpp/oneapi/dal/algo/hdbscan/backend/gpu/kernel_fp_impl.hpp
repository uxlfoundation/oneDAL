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

#pragma once

#include "oneapi/dal/algo/hdbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/hdbscan/backend/cluster_utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event kernels_fp<Float>::compute_squared_distances(sycl::queue& queue,
                                                         const pr::ndview<Float, 2>& data,
                                                         pr::ndview<Float, 2>& sq_dist,
                                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_squared_distances, queue);

    const std::int64_t n = data.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(sq_dist.get_dimension(0) == n);
    ONEDAL_ASSERT(sq_dist.get_dimension(1) == n);

    // Compute full pairwise squared L2 distance matrix using GEMM:
    // dist²[i,j] = ||x_i||² + ||x_j||² - 2 * x_i · x_j
    pr::squared_l2_distance<Float> dist_op(queue);
    return dist_op(data, data, sq_dist, deps);
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_core_distances(sycl::queue& queue,
                                                      const pr::ndview<Float, 2>& sq_dist,
                                                      pr::ndview<Float, 1>& core_distances,
                                                      std::int64_t min_samples,
                                                      std::int64_t row_count,
                                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(sq_dist.get_dimension(0) == n);
    ONEDAL_ASSERT(sq_dist.get_dimension(1) == n);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == n);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= n);

    // Select k-th nearest squared distance per row using kselect
    // Core distance index: min_samples - 1 (includes self at distance 0)
    // We select min_samples smallest values per row, then take the last one
    const std::int64_t k = min_samples;

    auto [ksel_vals, ksel_vals_event] =
        pr::ndarray<Float, 2>::zeros(queue, { n, k }, sycl::usm::alloc::device);

    pr::kselect_by_rows<Float> ksel(queue, { n, n }, k);
    bk::event_vector ksel_deps = deps;
    ksel_deps.push_back(ksel_vals_event);
    auto ksel_event = ksel(queue, sq_dist, k, ksel_vals, ksel_deps);

    // Extract the k-th value (last column) from kselect result and take sqrt
    // ksel_vals is [n x k], we need column index k-1 from each row
    const Float* ksel_ptr = ksel_vals.get_data();
    Float* core_ptr = core_distances.get_mutable_data();
    const std::int64_t kk = k;

    auto sqrt_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ksel_event });
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float sq_val = ksel_ptr[i * kk + (kk - 1)];
            core_ptr[i] = sycl::sqrt(sycl::fmax(sq_val, Float(0)));
        });
    });

    return sqrt_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_mrd_matrix(sycl::queue& queue,
                                                  const pr::ndview<Float, 1>& core_distances,
                                                  pr::ndview<Float, 2>& mrd_matrix,
                                                  const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_mrd_matrix, queue);

    const std::int64_t n = core_distances.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    // mrd_matrix already contains squared L2 distances from compute_squared_distances.
    // Transform in-place: mrd[i][j] = max(core_dist[i], core_dist[j], sqrt(sq_dist[i][j]))
    const Float* core_ptr = core_distances.get_data();
    Float* mrd_ptr = mrd_matrix.get_mutable_data();

    auto mrd_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            const Float eucl = sycl::sqrt(sycl::fmax(mrd_ptr[i * n + j], Float(0)));
            const Float cd_i = core_ptr[i];
            const Float cd_j = core_ptr[j];
            mrd_ptr[i * n + j] = sycl::fmax(sycl::fmax(cd_i, cd_j), eucl);
        });
    });

    return mrd_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::build_mst(sycl::queue& queue,
                                         const pr::ndview<Float, 2>& mrd_matrix,
                                         pr::ndview<std::int32_t, 1>& mst_from,
                                         pr::ndview<std::int32_t, 1>& mst_to,
                                         pr::ndview<Float, 1>& mst_weights,
                                         std::int64_t row_count,
                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst, queue);

    const std::int64_t n = row_count;
    const std::int64_t edge_count = n - 1;

    ONEDAL_ASSERT(n > 1);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);
    ONEDAL_ASSERT(mst_from.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_to.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_weights.get_dimension(0) == edge_count);

    // Prim's algorithm on the host using MRD data copied from device.
    // This avoids both the race condition in parallel atomic min-reduction
    // and the massive kernel launch overhead of per-iteration submission.
    // The MRD matrix is already computed on GPU; we copy it to host for
    // the inherently sequential MST construction, then copy results back.

    // Wait for MRD matrix and MST allocation to be ready
    sycl::event::wait(deps);

    // Copy MRD matrix to host
    auto mrd_host = mrd_matrix.to_host(queue);
    const Float* mrd_ptr = mrd_host.get_data();

    // Host-side Prim's algorithm (identical to cluster_utils.hpp::build_mst_on_host
    // but operates on the precomputed MRD matrix instead of raw data + core distances)
    std::vector<Float> min_edge_vec(n, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from_vec(n, 0);
    std::vector<bool> in_mst_vec(n, false);

    std::vector<std::int32_t> from_vec(edge_count);
    std::vector<std::int32_t> to_vec(edge_count);
    std::vector<Float> weight_vec(edge_count);

    in_mst_vec[0] = true;
    for (std::int64_t j = 1; j < n; j++) {
        min_edge_vec[j] = mrd_ptr[j]; // row 0, column j
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
        in_mst_vec[best] = true;

        for (std::int64_t j = 0; j < n; j++) {
            if (in_mst_vec[j])
                continue;
            const Float mrd = mrd_ptr[best * n + j];
            if (mrd < min_edge_vec[j]) {
                min_edge_vec[j] = mrd;
                min_from_vec[j] = best;
            }
        }
    }

    // Copy MST results back to device
    std::int32_t* from_ptr = mst_from.get_mutable_data();
    std::int32_t* to_ptr = mst_to.get_mutable_data();
    Float* weight_ptr = mst_weights.get_mutable_data();

    auto from_event = queue.memcpy(from_ptr, from_vec.data(), edge_count * sizeof(std::int32_t));
    auto to_event = queue.memcpy(to_ptr, to_vec.data(), edge_count * sizeof(std::int32_t));
    auto weight_event = queue.memcpy(weight_ptr, weight_vec.data(), edge_count * sizeof(Float));

    sycl::event::wait({ from_event, to_event, weight_event });

    return weight_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::sort_mst_by_weight(sycl::queue& queue,
                                                  pr::ndview<std::int32_t, 1>& mst_from,
                                                  pr::ndview<std::int32_t, 1>& mst_to,
                                                  pr::ndview<Float, 1>& mst_weights,
                                                  std::int64_t edge_count,
                                                  const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.sort_mst_by_weight, queue);

    ONEDAL_ASSERT(edge_count > 0);

    // Create index array [0, 1, 2, ..., edge_count-1]
    using Index = std::uint32_t;
    auto [indices, indices_event] =
        pr::ndarray<Index, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    Index* ind_ptr = indices.get_mutable_data();
    sycl::event ind_alloc_event = indices_event;
    auto iota_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.depends_on({ ind_alloc_event });
        h.parallel_for(sycl::range<1>(edge_count), [=](sycl::id<1> idx) {
            ind_ptr[idx[0]] = static_cast<Index>(idx[0]);
        });
    });

    // Sort weights with corresponding indices using radix sort
    pr::radix_sort_indices_inplace<Float, Index> sorter(queue);
    auto sort_event = sorter(mst_weights, indices, { iota_event });

    // Permute mst_from and mst_to according to sorted indices
    auto [sorted_from, sf_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [sorted_to, st_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    const std::int32_t* from_ptr = mst_from.get_data();
    const std::int32_t* to_ptr = mst_to.get_data();
    std::int32_t* sf_ptr = sorted_from.get_mutable_data();
    std::int32_t* st_ptr = sorted_to.get_mutable_data();
    const Index* sorted_ind_ptr = indices.get_data();

    sycl::event sf_alloc_event = sf_event;
    sycl::event st_alloc_event = st_event;
    auto permute_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ sort_event, sf_alloc_event, st_alloc_event });
        h.parallel_for(sycl::range<1>(edge_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Index orig = sorted_ind_ptr[i];
            sf_ptr[i] = from_ptr[orig];
            st_ptr[i] = to_ptr[orig];
        });
    });

    // Copy permuted results back into mst_from and mst_to
    std::int32_t* mst_from_ptr = mst_from.get_mutable_data();
    std::int32_t* mst_to_ptr = mst_to.get_mutable_data();

    auto copy_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ permute_event });
        h.parallel_for(sycl::range<1>(edge_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            mst_from_ptr[i] = sf_ptr[i];
            mst_to_ptr[i] = st_ptr[i];
        });
    });

    return copy_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::extract_clusters(sycl::queue& queue,
                                                const pr::ndview<std::int32_t, 1>& mst_from,
                                                const pr::ndview<std::int32_t, 1>& mst_to,
                                                const pr::ndview<Float, 1>& mst_weights,
                                                pr::ndview<std::int32_t, 1>& responses,
                                                std::int64_t row_count,
                                                std::int64_t min_cluster_size,
                                                const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.extract_clusters, queue);

    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_cluster_size >= 2);

    // The extract_clusters pipeline has both sequential and parallel phases.
    // We split them into separate kernels to maximize GPU utilization:
    //   Kernel 1 (single_task): Build Kruskal dendrogram via union-find
    //   Kernel 2 (parallel_for): Init dendrogram node arrays
    //   Kernel 3 (single_task): Build condensed tree + EOM stability selection
    //   Kernel 4 (parallel_for): Build dendro_parent array from dendrogram edges
    //   Kernel 5 (parallel_for): Label each point independently

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t total_nodes = 2 * row_count - 1;

    // Allocate working memory on device for the graph algorithm
    // Union-find arrays
    auto [uf_parent, uf_parent_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_size, comp_size_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_to_node, comp_to_node_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);

    // Dendrogram arrays
    auto [dendro_left, dl_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_right, dr_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_weight, dw_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_size, ds_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    // Condensed tree - worst case 2*row_count edges
    const std::int64_t max_condensed = 2 * row_count;
    auto [cond_parent, cp_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_child, cc_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_lambda, cl_event] =
        pr::ndarray<Float, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_size, cs_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_count_arr, cca_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    // Node-level arrays for condensed tree processing
    auto [node_size, ns_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [left_child, lc_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);
    auto [right_child, rc_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);
    auto [node_weight, nw_event] =
        pr::ndarray<Float, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [dendro_to_cluster, dtc_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);

    // EOM arrays (sized by max possible cluster count = total_nodes)
    const std::int64_t max_clusters = total_nodes;
    auto [stability, stab_event] =
        pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [lambda_birth, lb_event] =
        pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [is_leaf_cluster, ilc_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [is_selected, is_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [cluster_label, clab_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);
    auto [cluster_size_arr, csz_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [cluster_parent, cprt_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);

    // Child cluster lists - use flat arrays with max 2 children per cluster
    auto [child_clusters_0, cc0_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);
    auto [child_clusters_1, cc1_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);

    // Point labeling arrays
    auto [point_fell_from, pff_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    auto [dendro_parent, dp_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);

    // Stack for condensed tree building (node + cluster id)
    auto [stack_arr, stk_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [stack_cid, stk_cid_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);

    // Separate stack for leaf-collection DFS within collect_and_add_leaves.
    // Must not alias the condensed-tree stack (stack_arr/stack_cid) because
    // collect_and_add_leaves is called while the condensed-tree traversal is active.
    auto [leaf_stack_arr, leaf_stk_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);

    // n_clusters_arr: single-element device array to pass n_clusters from kernel 3 -> kernel 5
    auto [n_clusters_arr, nca_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    // C++17: structured bindings cannot be captured in SYCL lambdas.
    // Copy all allocation events to regular variables for lambda capture.
    sycl::event ev_uf_parent = uf_parent_event;
    sycl::event ev_comp_size = comp_size_event;
    sycl::event ev_comp_to_node = comp_to_node_event;
    sycl::event ev_dl = dl_event;
    sycl::event ev_dr = dr_event;
    sycl::event ev_dw = dw_event;
    sycl::event ev_ds = ds_event;
    sycl::event ev_ns = ns_event;
    sycl::event ev_lc = lc_event;
    sycl::event ev_rc = rc_event;
    sycl::event ev_nw = nw_event;
    sycl::event ev_dtc = dtc_event;
    sycl::event ev_cp = cp_event;
    sycl::event ev_cc = cc_event;
    sycl::event ev_cl = cl_event;
    sycl::event ev_cs = cs_event;
    sycl::event ev_cca = cca_event;
    sycl::event ev_stab = stab_event;
    sycl::event ev_lb = lb_event;
    sycl::event ev_ilc = ilc_event;
    sycl::event ev_is = is_event;
    sycl::event ev_clab = clab_event;
    sycl::event ev_csz = csz_event;
    sycl::event ev_cprt = cprt_event;
    sycl::event ev_cc0 = cc0_event;
    sycl::event ev_cc1 = cc1_event;
    sycl::event ev_pff = pff_event;
    sycl::event ev_dp = dp_event;
    sycl::event ev_stk = stk_event;
    sycl::event ev_stk_cid = stk_cid_event;
    sycl::event ev_leaf_stk = leaf_stk_event;
    sycl::event ev_nca = nca_event;

    // Capture all pointers for kernels
    const std::int32_t* mst_from_ptr = mst_from.get_data();
    const std::int32_t* mst_to_ptr = mst_to.get_data();
    const Float* mst_weights_ptr = mst_weights.get_data();
    std::int32_t* resp_ptr = responses.get_mutable_data();

    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* comp_size_ptr = comp_size.get_mutable_data();
    std::int32_t* comp_to_node_ptr = comp_to_node.get_mutable_data();

    std::int32_t* dl_ptr = dendro_left.get_mutable_data();
    std::int32_t* dr_ptr = dendro_right.get_mutable_data();
    Float* dw_ptr = dendro_weight.get_mutable_data();
    std::int32_t* dsz_ptr = dendro_size.get_mutable_data();

    std::int32_t* cond_p_ptr = cond_parent.get_mutable_data();
    std::int32_t* cond_c_ptr = cond_child.get_mutable_data();
    Float* cond_l_ptr = cond_lambda.get_mutable_data();
    std::int32_t* cond_s_ptr = cond_size.get_mutable_data();
    std::int32_t* cond_cnt_ptr = cond_count_arr.get_mutable_data();

    std::int32_t* ns_ptr = node_size.get_mutable_data();
    std::int32_t* lc_ptr = left_child.get_mutable_data();
    std::int32_t* rc_ptr = right_child.get_mutable_data();
    Float* nw_ptr = node_weight.get_mutable_data();
    std::int32_t* dtc_ptr = dendro_to_cluster.get_mutable_data();

    Float* stab_ptr = stability.get_mutable_data();
    Float* lb_ptr = lambda_birth.get_mutable_data();
    std::int32_t* ilc_ptr = is_leaf_cluster.get_mutable_data();
    std::int32_t* is_ptr = is_selected.get_mutable_data();
    std::int32_t* clab_ptr = cluster_label.get_mutable_data();
    std::int32_t* csz_ptr = cluster_size_arr.get_mutable_data();
    std::int32_t* cprt_ptr = cluster_parent.get_mutable_data();
    std::int32_t* cc0_ptr = child_clusters_0.get_mutable_data();
    std::int32_t* cc1_ptr = child_clusters_1.get_mutable_data();

    std::int32_t* pff_ptr = point_fell_from.get_mutable_data();
    std::int32_t* dp_ptr = dendro_parent.get_mutable_data();

    std::int32_t* stk_ptr = stack_arr.get_mutable_data();
    std::int32_t* stk_cid_ptr = stack_cid.get_mutable_data();
    std::int32_t* leaf_stk_ptr = leaf_stack_arr.get_mutable_data();

    std::int32_t* nc_ptr = n_clusters_arr.get_mutable_data();

    const std::int64_t rc = row_count;
    const std::int64_t ec = edge_count;
    const std::int64_t mcs = min_cluster_size;
    const std::int64_t tn = total_nodes;

    // =========================================================================
    // Kernel 1 (single_task): Build Kruskal dendrogram via union-find
    // Sequential — edges must be processed in sorted order
    // =========================================================================
    bk::event_vector k1_deps = deps;
    k1_deps.insert(k1_deps.end(),
                   { ev_uf_parent, ev_comp_size, ev_comp_to_node, ev_dl, ev_dr, ev_dw, ev_ds });

    auto k1_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(k1_deps);
        h.single_task([=]() {
            for (std::int64_t i = 0; i < rc; i++) {
                uf_parent_ptr[i] = static_cast<std::int32_t>(i);
                comp_size_ptr[i] = 1;
                comp_to_node_ptr[i] = static_cast<std::int32_t>(i);
            }

            auto uf_find = [&](std::int32_t x) -> std::int32_t {
                while (uf_parent_ptr[x] != x) {
                    uf_parent_ptr[x] = uf_parent_ptr[uf_parent_ptr[x]];
                    x = uf_parent_ptr[x];
                }
                return x;
            };

            for (std::int64_t e = 0; e < ec; e++) {
                const std::int32_t ru = uf_find(mst_from_ptr[e]);
                const std::int32_t rv = uf_find(mst_to_ptr[e]);
                if (ru == rv)
                    continue;

                const std::int32_t left_node = comp_to_node_ptr[ru];
                const std::int32_t right_node = comp_to_node_ptr[rv];
                const std::int32_t new_size = comp_size_ptr[ru] + comp_size_ptr[rv];

                dl_ptr[e] = left_node;
                dr_ptr[e] = right_node;
                dw_ptr[e] = mst_weights_ptr[e];
                dsz_ptr[e] = new_size;

                if (comp_size_ptr[ru] < comp_size_ptr[rv]) {
                    uf_parent_ptr[ru] = rv;
                    comp_size_ptr[rv] = new_size;
                    comp_to_node_ptr[rv] = static_cast<std::int32_t>(rc + e);
                }
                else {
                    uf_parent_ptr[rv] = ru;
                    comp_size_ptr[ru] = new_size;
                    comp_to_node_ptr[ru] = static_cast<std::int32_t>(rc + e);
                }
            }
        });
    });

    // =========================================================================
    // Kernel 2 (parallel_for): Initialize node arrays from dendrogram
    // Each work-item handles one dendrogram node or one leaf node
    // =========================================================================
    auto k2_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k1_event, ev_ns, ev_lc, ev_rc, ev_nw });
        h.parallel_for(sycl::range<1>(total_nodes), [=](sycl::id<1> idx) {
            const std::int64_t nid = idx[0];
            if (nid < rc) {
                // Leaf node
                ns_ptr[nid] = 1;
            }
            else {
                // Internal dendrogram node
                const std::int64_t e = nid - rc;
                ns_ptr[nid] = dsz_ptr[e];
                lc_ptr[nid] = dl_ptr[e];
                rc_ptr[nid] = dr_ptr[e];
                nw_ptr[nid] = dw_ptr[e];
            }
        });
    });

    // =========================================================================
    // Kernel 3 (single_task): Build condensed tree + EOM selection + mappings
    // Sequential — graph traversal, stability computation, cluster selection
    // =========================================================================
    bk::event_vector k3_deps = {
        k2_event, ev_dtc, ev_cp,  ev_cc,  ev_cl,      ev_cs,       ev_cca,
        ev_stab,  ev_lb,  ev_ilc, ev_is,  ev_clab,    ev_csz,      ev_cprt,
        ev_cc0,   ev_cc1, ev_pff, ev_stk, ev_stk_cid, ev_leaf_stk, ev_nca
    };

    auto k3_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(k3_deps);
        h.single_task([=]() {
            // Find root
            std::int32_t root = -1;
            for (std::int64_t e = ec - 1; e >= 0; e--) {
                if (dsz_ptr[e] > 0) {
                    root = static_cast<std::int32_t>(rc + e);
                    break;
                }
            }

            if (root < 0) {
                for (std::int64_t i = 0; i < rc; i++)
                    resp_ptr[i] = -1;
                nc_ptr[0] = 0;
                return;
            }

            std::int32_t next_cid = static_cast<std::int32_t>(rc);
            dtc_ptr[root] = next_cid++;

            cond_cnt_ptr[0] = 0;

            auto add_condensed_edge =
                [&](std::int32_t parent, std::int32_t child, Float lambda, std::int32_t child_sz) {
                    const std::int32_t idx = cond_cnt_ptr[0]++;
                    cond_p_ptr[idx] = parent;
                    cond_c_ptr[idx] = child;
                    cond_l_ptr[idx] = lambda;
                    cond_s_ptr[idx] = child_sz;
                };

            // Iterative collect_leaves using a separate stack (leaf_stk_ptr)
            // to avoid corrupting the condensed-tree traversal stack.
            auto collect_and_add_leaves =
                [&](std::int32_t subtree_root, std::int32_t parent_cid, Float lambda) {
                    std::int32_t sp = 0;
                    leaf_stk_ptr[sp++] = subtree_root;
                    while (sp > 0) {
                        const std::int32_t nd = leaf_stk_ptr[--sp];
                        if (nd < rc) {
                            add_condensed_edge(parent_cid, nd, lambda, 1);
                        }
                        else {
                            if (lc_ptr[nd] >= 0)
                                leaf_stk_ptr[sp++] = lc_ptr[nd];
                            if (rc_ptr[nd] >= 0)
                                leaf_stk_ptr[sp++] = rc_ptr[nd];
                        }
                    }
                };

            // Build condensed tree using explicit stack
            std::int32_t ct_sp = 0;
            stk_ptr[ct_sp] = root;
            stk_cid_ptr[ct_sp] = dtc_ptr[root];
            ct_sp++;

            while (ct_sp > 0) {
                ct_sp--;
                const std::int32_t nid = stk_ptr[ct_sp];
                const std::int32_t parent_cid = stk_cid_ptr[ct_sp];

                if (nid < rc)
                    continue;

                const std::int32_t lc_node = lc_ptr[nid];
                const std::int32_t rc_node = rc_ptr[nid];
                if (lc_node < 0 || rc_node < 0)
                    continue;

                const std::int32_t ls = ns_ptr[lc_node];
                const std::int32_t rs = ns_ptr[rc_node];
                const Float w = nw_ptr[nid];
                const Float lambda = (w > Float(0)) ? Float(1) / w : Float(1e30);

                const bool l_big = ls >= mcs;
                const bool r_big = rs >= mcs;

                if (l_big && r_big) {
                    const std::int32_t lcid = next_cid++;
                    const std::int32_t rcid = next_cid++;
                    dtc_ptr[lc_node] = lcid;
                    dtc_ptr[rc_node] = rcid;
                    add_condensed_edge(parent_cid, lcid, lambda, ls);
                    add_condensed_edge(parent_cid, rcid, lambda, rs);
                    stk_ptr[ct_sp] = lc_node;
                    stk_cid_ptr[ct_sp] = lcid;
                    ct_sp++;
                    stk_ptr[ct_sp] = rc_node;
                    stk_cid_ptr[ct_sp] = rcid;
                    ct_sp++;
                }
                else if (l_big) {
                    dtc_ptr[lc_node] = parent_cid;
                    collect_and_add_leaves(rc_node, parent_cid, lambda);
                    stk_ptr[ct_sp] = lc_node;
                    stk_cid_ptr[ct_sp] = parent_cid;
                    ct_sp++;
                }
                else if (r_big) {
                    dtc_ptr[rc_node] = parent_cid;
                    collect_and_add_leaves(lc_node, parent_cid, lambda);
                    stk_ptr[ct_sp] = rc_node;
                    stk_cid_ptr[ct_sp] = parent_cid;
                    ct_sp++;
                }
                else {
                    collect_and_add_leaves(lc_node, parent_cid, lambda);
                    collect_and_add_leaves(rc_node, parent_cid, lambda);
                }
            }

            // --- EOM: Compute stability and select clusters ---
            const std::int32_t n_clusters = next_cid;
            const std::int32_t root_cid = static_cast<std::int32_t>(rc);
            const std::int32_t cond_count = cond_cnt_ptr[0];

            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                ilc_ptr[c] = 1;
                is_ptr[c] = 1;
            }
            csz_ptr[root_cid] = static_cast<std::int32_t>(rc);

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (cond_c_ptr[i] >= rc) {
                    lb_ptr[cond_c_ptr[i]] = cond_l_ptr[i];
                    ilc_ptr[cond_p_ptr[i]] = 0;
                    if (cc0_ptr[cond_p_ptr[i]] < 0)
                        cc0_ptr[cond_p_ptr[i]] = cond_c_ptr[i];
                    else
                        cc1_ptr[cond_p_ptr[i]] = cond_c_ptr[i];
                    csz_ptr[cond_c_ptr[i]] = cond_s_ptr[i];
                }
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                const Float birth = lb_ptr[cond_p_ptr[i]];
                const Float contrib = (cond_l_ptr[i] - birth) * static_cast<Float>(cond_s_ptr[i]);
                if (contrib > Float(0))
                    stab_ptr[cond_p_ptr[i]] += contrib;
            }

            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                if (csz_ptr[c] < mcs)
                    is_ptr[c] = 0;
            }

            // Bottom-up EOM
            for (std::int32_t c = n_clusters - 1; c >= root_cid; c--) {
                if (ilc_ptr[c])
                    continue;

                Float child_sum = Float(0);
                if (cc0_ptr[c] >= 0)
                    child_sum += stab_ptr[cc0_ptr[c]];
                if (cc1_ptr[c] >= 0)
                    child_sum += stab_ptr[cc1_ptr[c]];

                if (child_sum > stab_ptr[c]) {
                    is_ptr[c] = 0;
                    stab_ptr[c] = child_sum;
                }
                else {
                    std::int32_t dsp = 0;
                    if (cc0_ptr[c] >= 0)
                        stk_ptr[dsp++] = cc0_ptr[c];
                    if (cc1_ptr[c] >= 0)
                        stk_ptr[dsp++] = cc1_ptr[c];
                    while (dsp > 0) {
                        const std::int32_t d = stk_ptr[--dsp];
                        is_ptr[d] = 0;
                        if (cc0_ptr[d] >= 0)
                            stk_ptr[dsp++] = cc0_ptr[d];
                        if (cc1_ptr[d] >= 0)
                            stk_ptr[dsp++] = cc1_ptr[d];
                    }
                }
            }

            // --- Assign cluster labels and build mappings ---
            std::int32_t label_counter = 0;
            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                if (is_ptr[c])
                    clab_ptr[c] = label_counter++;
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (cond_c_ptr[i] >= rc)
                    cprt_ptr[cond_c_ptr[i]] = cond_p_ptr[i];
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (cond_c_ptr[i] < rc)
                    pff_ptr[cond_c_ptr[i]] = cond_p_ptr[i];
            }

            // Store n_clusters for kernel 5
            nc_ptr[0] = n_clusters;
        });
    });

    // =========================================================================
    // Kernel 4 (parallel_for): Build dendro_parent from dendrogram edges
    // Each work-item processes one dendrogram edge independently
    // =========================================================================
    auto k4_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k3_event, ev_dp });
        h.parallel_for(sycl::range<1>(edge_count), [=](sycl::id<1> idx) {
            const std::int64_t e = idx[0];
            if (dsz_ptr[e] > 0) {
                const std::int64_t nid = rc + e;
                dp_ptr[dl_ptr[e]] = static_cast<std::int32_t>(nid);
                dp_ptr[dr_ptr[e]] = static_cast<std::int32_t>(nid);
            }
        });
    });

    // =========================================================================
    // Kernel 5 (parallel_for): Label each point independently
    // Each work-item walks up cluster tree or dendrogram tree for one point
    // =========================================================================
    auto k5_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k4_event });
        h.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t n_clusters = nc_ptr[0];

            // If n_clusters == 0, kernel 3 already set resp_ptr[i] = -1
            if (n_clusters == 0)
                return;

            const std::int32_t root_cid = static_cast<std::int32_t>(rc);
            resp_ptr[i] = -1;

            // Case 1: Point fell out of a cluster — walk up cluster tree
            const std::int32_t fell = pff_ptr[i];
            if (fell >= 0) {
                std::int32_t c = fell;
                while (c >= root_cid && c < n_clusters) {
                    if (is_ptr[c]) {
                        resp_ptr[i] = clab_ptr[c];
                        return;
                    }
                    c = cprt_ptr[c];
                }
                return;
            }

            // Case 2: Point was never ejected — walk up dendrogram tree
            std::int32_t nid = static_cast<std::int32_t>(i);
            while (nid >= 0 && nid < static_cast<std::int32_t>(tn)) {
                const std::int32_t cid = dtc_ptr[nid];
                if (cid >= root_cid) {
                    std::int32_t c = cid;
                    while (c >= root_cid && c < n_clusters) {
                        if (is_ptr[c]) {
                            resp_ptr[i] = clab_ptr[c];
                            return;
                        }
                        c = cprt_ptr[c];
                    }
                    return;
                }
                nid = dp_ptr[nid];
            }
        });
    });

    return k5_event;
}

#endif

} // namespace oneapi::dal::hdbscan::backend
