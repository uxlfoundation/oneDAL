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

/// Working pointers and constants for cluster extraction kernels.
/// Groups all device pointers and scalar parameters so they can be
/// passed to helper functions without 30+ individual parameters.
template <typename Float>
struct cluster_work_ptrs {
    const std::int32_t* mst_from_ptr;
    const std::int32_t* mst_to_ptr;
    const Float* mst_weights_ptr;
    std::int32_t* resp_ptr;

    std::int32_t* uf_parent_ptr;
    std::int32_t* comp_size_ptr;
    std::int32_t* comp_to_node_ptr;

    std::int32_t* dl_ptr;
    std::int32_t* dr_ptr;
    Float* dw_ptr;
    std::int32_t* dsz_ptr;

    std::int32_t* ns_ptr;
    std::int32_t* lc_ptr;
    std::int32_t* rc_ptr;
    Float* nw_ptr;
    std::int32_t* dtc_ptr;

    std::int32_t* cond_p_ptr;
    std::int32_t* cond_c_ptr;
    Float* cond_l_ptr;
    std::int32_t* cond_s_ptr;
    std::int32_t* cond_cnt_ptr;

    Float* stab_ptr;
    Float* lb_ptr;
    std::int32_t* ilc_ptr;
    std::int32_t* is_ptr;
    std::int32_t* clab_ptr;
    std::int32_t* csz_ptr;
    std::int32_t* cprt_ptr;
    std::int32_t* cc0_ptr;
    std::int32_t* cc1_ptr;

    std::int32_t* pff_ptr;
    std::int32_t* dp_ptr;

    std::int32_t* stk_ptr;
    std::int32_t* stk_cid_ptr;
    std::int32_t* leaf_stk_ptr;

    std::int32_t* nc_ptr;

    std::int64_t row_count;
    std::int64_t edge_count;
    std::int64_t min_cluster_size;
    std::int64_t total_nodes;
    std::int32_t cluster_selection; // 0 = EOM, 1 = leaf
    bool allow_single_cluster;
    double cluster_selection_epsilon;
    std::int64_t max_cluster_size;
};

template <typename Float>
sycl::event kernels_fp<Float>::compute_distance_matrix(sycl::queue& queue,
                                                       const pr::ndview<Float, 2>& data,
                                                       pr::ndview<Float, 2>& dist,
                                                       distance_metric metric,
                                                       double degree,
                                                       const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_distance_matrix, queue);

    const std::int64_t n = data.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);

    switch (metric) {
        case distance_metric::cosine: {
            pr::cosine_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
        case distance_metric::manhattan: {
            pr::distance<Float, pr::lp_metric<Float>> dist_op(queue,
                                                              pr::lp_metric<Float>(Float(1)));
            return dist_op(data, data, dist, deps);
        }
        case distance_metric::minkowski: {
            pr::distance<Float, pr::lp_metric<Float>> dist_op(
                queue,
                pr::lp_metric<Float>(static_cast<Float>(degree)));
            return dist_op(data, data, dist, deps);
        }
        case distance_metric::chebyshev: {
            pr::chebyshev_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
        default: { // euclidean — compute squared L2 (sqrt applied later)
            pr::squared_l2_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
    }
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_core_distances(sycl::queue& queue,
                                                      const pr::ndview<Float, 2>& dist,
                                                      pr::ndview<Float, 1>& core_distances,
                                                      std::int64_t min_samples,
                                                      std::int64_t row_count,
                                                      distance_metric metric,
                                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == n);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= n);

    const std::int64_t k = min_samples;

    auto [ksel_vals, ksel_vals_event] =
        pr::ndarray<Float, 2>::zeros(queue, { n, k }, sycl::usm::alloc::device);

    pr::kselect_by_rows<Float> ksel(queue, { n, n }, k);
    bk::event_vector ksel_deps = deps;
    ksel_deps.push_back(ksel_vals_event);
    auto ksel_event = ksel(queue, dist, k, ksel_vals, ksel_deps);

    const Float* ksel_ptr = ksel_vals.get_data();
    Float* core_ptr = core_distances.get_mutable_data();
    const std::int64_t kk = k;
    const bool needs_sqrt = (metric == distance_metric::euclidean);

    auto extract_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ksel_event });
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float val = ksel_ptr[i * kk + (kk - 1)];
            core_ptr[i] = needs_sqrt ? sycl::sqrt(sycl::fmax(val, Float(0))) : val;
        });
    });

    return extract_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_mrd_matrix(sycl::queue& queue,
                                                  const pr::ndview<Float, 1>& core_distances,
                                                  pr::ndview<Float, 2>& mrd_matrix,
                                                  distance_metric metric,
                                                  const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_mrd_matrix, queue);

    const std::int64_t n = core_distances.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    const Float* core_ptr = core_distances.get_data();
    Float* mrd_ptr = mrd_matrix.get_mutable_data();
    const bool needs_sqrt = (metric == distance_metric::euclidean);

    auto mrd_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            // For euclidean: mrd_matrix has squared L2, need sqrt first
            // For other metrics: mrd_matrix has actual distances
            const Float d = needs_sqrt ? sycl::sqrt(sycl::fmax(mrd_ptr[i * n + j], Float(0)))
                                       : mrd_ptr[i * n + j];
            const Float cd_i = core_ptr[i];
            const Float cd_j = core_ptr[j];
            mrd_ptr[i * n + j] = sycl::fmax(sycl::fmax(cd_i, cd_j), d);
        });
    });

    return mrd_event;
}

/// GPU Boruvka helper: each work-item finds the nearest neighbor in a
/// different component by scanning one row of the precomputed MRD matrix.
/// Writes per-point best MRD and best neighbor index.
template <typename Float>
static sycl::event boruvka_find_nearest_mrd(sycl::queue& queue,
                                            const Float* mrd_ptr,
                                            const std::int32_t* comp_ptr,
                                            Float* pt_best_mrd_ptr,
                                            std::int32_t* pt_best_idx_ptr,
                                            std::int64_t n,
                                            const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t my_comp = comp_ptr[i];
            const Float* row = mrd_ptr + i * n;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_ptr[j] != my_comp && row[j] < best_m) {
                    best_m = row[j];
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd_ptr[i] = best_m;
            pt_best_idx_ptr[i] = best_j;
        });
    });
}

/// GPU Boruvka helper: each work-item finds the nearest neighbor in a
/// different component using on-the-fly MRD distance computation.
template <typename Float>
static sycl::event boruvka_find_nearest_otf(sycl::queue& queue,
                                            const Float* data_ptr,
                                            std::int64_t col_count,
                                            const Float* core_ptr,
                                            const std::int32_t* comp_ptr,
                                            Float* pt_best_mrd_ptr,
                                            std::int32_t* pt_best_idx_ptr,
                                            std::int64_t n,
                                            std::int32_t metric_id,
                                            Float degree,
                                            const bk::event_vector& deps) {
    const std::int64_t d = col_count;
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t my_comp = comp_ptr[i];
            const Float my_core = core_ptr[i];
            const Float* xi = data_ptr + i * d;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_ptr[j] == my_comp)
                    continue;
                const Float* xj = data_ptr + j * d;
                Float dist = Float(0);
                if (metric_id == 1) { // manhattan
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = xi[dd] - xj[dd];
                        dist += sycl::fabs(diff);
                    }
                }
                else if (metric_id == 2) { // minkowski
                    Float sum = Float(0);
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = sycl::fabs(xi[dd] - xj[dd]);
                        sum += sycl::pow(diff, degree);
                    }
                    dist = sycl::pow(sum, Float(1) / degree);
                }
                else if (metric_id == 3) { // chebyshev
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = sycl::fabs(xi[dd] - xj[dd]);
                        if (diff > dist)
                            dist = diff;
                    }
                }
                else { // euclidean
                    Float sum = Float(0);
                    for (std::int64_t dd = 0; dd < d; dd++) {
                        Float diff = xi[dd] - xj[dd];
                        sum += diff * diff;
                    }
                    dist = sycl::sqrt(sycl::fmax(sum, Float(0)));
                }
                Float mrd = sycl::fmax(sycl::fmax(my_core, core_ptr[j]), dist);
                if (mrd < best_m) {
                    best_m = mrd;
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd_ptr[i] = best_m;
            pt_best_idx_ptr[i] = best_j;
        });
    });
}

/// GPU Boruvka: reduce per-point bests to per-component bests, merge components,
/// add MST edges. Runs as single_task because the merge step has data dependencies
/// (union-find), but operates on O(N) data — not the O(N²) distance matrix.
template <typename Float>
static sycl::event boruvka_merge_components(sycl::queue& queue,
                                            std::int32_t* comp_ptr,
                                            std::int32_t* uf_parent_ptr,
                                            std::int32_t* uf_rank_ptr,
                                            const Float* pt_best_mrd_ptr,
                                            const std::int32_t* pt_best_idx_ptr,
                                            Float* comp_best_mrd_ptr,
                                            std::int32_t* comp_best_from_ptr,
                                            std::int32_t* comp_best_to_ptr,
                                            std::int32_t* mst_from_ptr,
                                            std::int32_t* mst_to_ptr,
                                            Float* mst_weight_ptr,
                                            std::int32_t* edges_added_ptr,
                                            std::int32_t* num_comp_ptr,
                                            std::int64_t n,
                                            const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            const Float inf = std::numeric_limits<Float>::max();

            // Reset per-component best
            for (std::int64_t c = 0; c < n; c++) {
                comp_best_mrd_ptr[c] = inf;
                comp_best_from_ptr[c] = -1;
                comp_best_to_ptr[c] = -1;
            }

            // Reduce per-point bests to per-component bests
            for (std::int64_t i = 0; i < n; i++) {
                if (pt_best_idx_ptr[i] < 0)
                    continue;
                const std::int32_t c = comp_ptr[i];
                if (pt_best_mrd_ptr[i] < comp_best_mrd_ptr[c]) {
                    comp_best_mrd_ptr[c] = pt_best_mrd_ptr[i];
                    comp_best_from_ptr[c] = static_cast<std::int32_t>(i);
                    comp_best_to_ptr[c] = pt_best_idx_ptr[i];
                }
            }

            auto uf_find = [&](std::int32_t x) -> std::int32_t {
                while (uf_parent_ptr[x] != x) {
                    uf_parent_ptr[x] = uf_parent_ptr[uf_parent_ptr[x]];
                    x = uf_parent_ptr[x];
                }
                return x;
            };
            auto uf_union = [&](std::int32_t rx, std::int32_t ry) {
                if (uf_rank_ptr[rx] < uf_rank_ptr[ry])
                    uf_parent_ptr[rx] = ry;
                else if (uf_rank_ptr[rx] > uf_rank_ptr[ry])
                    uf_parent_ptr[ry] = rx;
                else {
                    uf_parent_ptr[ry] = rx;
                    uf_rank_ptr[rx]++;
                }
            };

            std::int32_t ea = edges_added_ptr[0];
            std::int32_t added = 0;
            for (std::int64_t c = 0; c < n; c++) {
                if (comp_best_from_ptr[c] < 0)
                    continue;
                const std::int32_t u = comp_best_from_ptr[c];
                const std::int32_t v = comp_best_to_ptr[c];
                const std::int32_t ru = uf_find(u), rv = uf_find(v);
                if (ru == rv)
                    continue;
                mst_from_ptr[ea] = u;
                mst_to_ptr[ea] = v;
                mst_weight_ptr[ea] = comp_best_mrd_ptr[c];
                ea++;
                added++;
                uf_union(ru, rv);
            }
            edges_added_ptr[0] = ea;
            num_comp_ptr[0] -= added;
        });
    });
}

/// GPU Boruvka: path-compress component IDs in parallel.
template <typename Float>
static sycl::event boruvka_compress_components(sycl::queue& queue,
                                               std::int32_t* comp_ptr,
                                               const std::int32_t* uf_parent_ptr,
                                               std::int64_t n,
                                               const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            std::int32_t x = static_cast<std::int32_t>(idx[0]);
            while (uf_parent_ptr[x] != x)
                x = uf_parent_ptr[x];
            comp_ptr[idx[0]] = x;
        });
    });
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

    ONEDAL_ASSERT(n > 1);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    // Allocate working arrays on device
    auto [comp, comp_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_parent, uf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_rank, ur_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_mrd, pbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_idx, pbi_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_mrd, cbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_from, cbf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_to, cbt_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(
        init_deps.end(),
        { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, cbt_ev, ea_ev, nc_ev });

    // Initialize comp[i] = i, uf_parent[i] = i, num_comp = n
    auto init_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(init_deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            comp_ptr[i] = i;
            uf_parent_ptr[i] = i;
            if (i == 0)
                num_comp_ptr[0] = static_cast<std::int32_t>(n);
        });
    });

    const Float* mrd_ptr = mrd_matrix.get_data();
    const std::int32_t max_rounds = 64;
    sycl::event last_event = init_event;

    for (std::int32_t round = 0; round < max_rounds; round++) {
        // Step A: parallel find nearest different-component neighbor
        auto find_event = boruvka_find_nearest_mrd(queue,
                                                   mrd_ptr,
                                                   comp_ptr,
                                                   pt_best_mrd.get_mutable_data(),
                                                   pt_best_idx.get_mutable_data(),
                                                   n,
                                                   { last_event });

        // Step B: reduce + merge (single_task — O(N) work)
        auto merge_event = boruvka_merge_components(queue,
                                                    comp_ptr,
                                                    uf_parent_ptr,
                                                    uf_rank_ptr,
                                                    pt_best_mrd.get_data(),
                                                    pt_best_idx.get_data(),
                                                    comp_best_mrd.get_mutable_data(),
                                                    comp_best_from.get_mutable_data(),
                                                    comp_best_to.get_mutable_data(),
                                                    mst_from.get_mutable_data(),
                                                    mst_to.get_mutable_data(),
                                                    mst_weights.get_mutable_data(),
                                                    edges_added_arr.get_mutable_data(),
                                                    num_comp_arr.get_mutable_data(),
                                                    n,
                                                    { find_event });

        // Step C: parallel path compression
        auto compress_event =
            boruvka_compress_components<Float>(queue, comp_ptr, uf_parent_ptr, n, { merge_event });

        // Check termination
        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        if (num_comp_host.get_data()[0] <= 1) {
            last_event = compress_event;
            break;
        }

        last_event = compress_event;
    }

    return last_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::build_mst_otf(sycl::queue& queue,
                                             const pr::ndview<Float, 2>& data,
                                             const pr::ndview<Float, 1>& core_distances,
                                             pr::ndview<std::int32_t, 1>& mst_from,
                                             pr::ndview<std::int32_t, 1>& mst_to,
                                             pr::ndview<Float, 1>& mst_weights,
                                             std::int64_t row_count,
                                             std::int64_t col_count,
                                             distance_metric metric,
                                             double degree,
                                             const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst_otf, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 1);

    std::int32_t metric_id = 0; // euclidean
    if (metric == distance_metric::manhattan)
        metric_id = 1;
    else if (metric == distance_metric::minkowski)
        metric_id = 2;
    else if (metric == distance_metric::chebyshev)
        metric_id = 3;

    // Allocate working arrays on device
    auto [comp, comp_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_parent, uf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [uf_rank, ur_ev] = pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_mrd, pbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [pt_best_idx, pbi_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_mrd, cbm_ev] = pr::ndarray<Float, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_from, cbf_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [comp_best_to, cbt_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, n, sycl::usm::alloc::device);
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(
        init_deps.end(),
        { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, cbt_ev, ea_ev, nc_ev });

    auto init_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(init_deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            comp_ptr[i] = i;
            uf_parent_ptr[i] = i;
            if (i == 0)
                num_comp_ptr[0] = static_cast<std::int32_t>(n);
        });
    });

    const Float* data_ptr = data.get_data();
    const Float* core_ptr = core_distances.get_data();
    const Float deg_f = static_cast<Float>(degree);
    const std::int32_t max_rounds = 64;
    sycl::event last_event = init_event;

    for (std::int32_t round = 0; round < max_rounds; round++) {
        auto find_event = boruvka_find_nearest_otf(queue,
                                                   data_ptr,
                                                   col_count,
                                                   core_ptr,
                                                   comp_ptr,
                                                   pt_best_mrd.get_mutable_data(),
                                                   pt_best_idx.get_mutable_data(),
                                                   n,
                                                   metric_id,
                                                   deg_f,
                                                   { last_event });

        auto merge_event = boruvka_merge_components(queue,
                                                    comp_ptr,
                                                    uf_parent_ptr,
                                                    uf_rank_ptr,
                                                    pt_best_mrd.get_data(),
                                                    pt_best_idx.get_data(),
                                                    comp_best_mrd.get_mutable_data(),
                                                    comp_best_from.get_mutable_data(),
                                                    comp_best_to.get_mutable_data(),
                                                    mst_from.get_mutable_data(),
                                                    mst_to.get_mutable_data(),
                                                    mst_weights.get_mutable_data(),
                                                    edges_added_arr.get_mutable_data(),
                                                    num_comp_arr.get_mutable_data(),
                                                    n,
                                                    { find_event });

        auto compress_event =
            boruvka_compress_components<Float>(queue, comp_ptr, uf_parent_ptr, n, { merge_event });

        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        if (num_comp_host.get_data()[0] <= 1) {
            last_event = compress_event;
            break;
        }

        last_event = compress_event;
    }

    return last_event;
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

// =========================================================================
// Cluster extraction helper: Kernels 1+2
// Build Kruskal dendrogram via union-find, then init node arrays.
// =========================================================================
template <typename Float>
sycl::event build_dendrogram_kernels(sycl::queue& queue,
                                     const cluster_work_ptrs<Float>& w,
                                     const bk::event_vector& deps) {
    // Kernel 1 (single_task): Build Kruskal dendrogram via union-find
    // Sequential — edges must be processed in sorted order
    auto k1_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            for (std::int64_t i = 0; i < w.row_count; i++) {
                w.uf_parent_ptr[i] = static_cast<std::int32_t>(i);
                w.comp_size_ptr[i] = 1;
                w.comp_to_node_ptr[i] = static_cast<std::int32_t>(i);
            }

            auto uf_find = [&](std::int32_t x) -> std::int32_t {
                while (w.uf_parent_ptr[x] != x) {
                    w.uf_parent_ptr[x] = w.uf_parent_ptr[w.uf_parent_ptr[x]];
                    x = w.uf_parent_ptr[x];
                }
                return x;
            };

            for (std::int64_t e = 0; e < w.edge_count; e++) {
                const std::int32_t ru = uf_find(w.mst_from_ptr[e]);
                const std::int32_t rv = uf_find(w.mst_to_ptr[e]);
                if (ru == rv)
                    continue;

                const std::int32_t left_node = w.comp_to_node_ptr[ru];
                const std::int32_t right_node = w.comp_to_node_ptr[rv];
                const std::int32_t new_size = w.comp_size_ptr[ru] + w.comp_size_ptr[rv];

                w.dl_ptr[e] = left_node;
                w.dr_ptr[e] = right_node;
                w.dw_ptr[e] = w.mst_weights_ptr[e];
                w.dsz_ptr[e] = new_size;

                if (w.comp_size_ptr[ru] < w.comp_size_ptr[rv]) {
                    w.uf_parent_ptr[ru] = rv;
                    w.comp_size_ptr[rv] = new_size;
                    w.comp_to_node_ptr[rv] = static_cast<std::int32_t>(w.row_count + e);
                }
                else {
                    w.uf_parent_ptr[rv] = ru;
                    w.comp_size_ptr[ru] = new_size;
                    w.comp_to_node_ptr[ru] = static_cast<std::int32_t>(w.row_count + e);
                }
            }
        });
    });

    // Kernel 2 (parallel_for): Initialize node arrays from dendrogram
    // Each work-item handles one dendrogram node or one leaf node
    auto k2_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k1_event });
        h.parallel_for(sycl::range<1>(w.total_nodes), [=](sycl::id<1> idx) {
            const std::int64_t nid = idx[0];
            if (nid < w.row_count) {
                w.ns_ptr[nid] = 1;
            }
            else {
                const std::int64_t e = nid - w.row_count;
                w.ns_ptr[nid] = w.dsz_ptr[e];
                w.lc_ptr[nid] = w.dl_ptr[e];
                w.rc_ptr[nid] = w.dr_ptr[e];
                w.nw_ptr[nid] = w.dw_ptr[e];
            }
        });
    });

    return k2_event;
}

// =========================================================================
// Cluster extraction helper: Kernel 3a
// Build the condensed tree from the dendrogram.
// =========================================================================
template <typename Float>
sycl::event build_condensed_tree_kernel(sycl::queue& queue,
                                        const cluster_work_ptrs<Float>& w,
                                        const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            std::int32_t root = -1;
            for (std::int64_t e = w.edge_count - 1; e >= 0; e--) {
                if (w.dsz_ptr[e] > 0) {
                    root = static_cast<std::int32_t>(w.row_count + e);
                    break;
                }
            }

            if (root < 0) {
                for (std::int64_t i = 0; i < w.row_count; i++)
                    w.resp_ptr[i] = -1;
                w.nc_ptr[0] = 0;
                return;
            }

            std::int32_t next_cid = static_cast<std::int32_t>(w.row_count);
            w.dtc_ptr[root] = next_cid++;

            w.cond_cnt_ptr[0] = 0;

            auto add_condensed_edge =
                [&](std::int32_t parent, std::int32_t child, Float lambda, std::int32_t child_sz) {
                    const std::int32_t idx = w.cond_cnt_ptr[0]++;
                    w.cond_p_ptr[idx] = parent;
                    w.cond_c_ptr[idx] = child;
                    w.cond_l_ptr[idx] = lambda;
                    w.cond_s_ptr[idx] = child_sz;
                };

            auto collect_and_add_leaves =
                [&](std::int32_t subtree_root, std::int32_t parent_cid, Float lambda) {
                    std::int32_t sp = 0;
                    w.leaf_stk_ptr[sp++] = subtree_root;
                    while (sp > 0) {
                        const std::int32_t nd = w.leaf_stk_ptr[--sp];
                        if (nd < w.row_count) {
                            add_condensed_edge(parent_cid, nd, lambda, 1);
                        }
                        else {
                            if (w.lc_ptr[nd] >= 0)
                                w.leaf_stk_ptr[sp++] = w.lc_ptr[nd];
                            if (w.rc_ptr[nd] >= 0)
                                w.leaf_stk_ptr[sp++] = w.rc_ptr[nd];
                        }
                    }
                };

            std::int32_t ct_sp = 0;
            w.stk_ptr[ct_sp] = root;
            w.stk_cid_ptr[ct_sp] = w.dtc_ptr[root];
            ct_sp++;

            while (ct_sp > 0) {
                ct_sp--;
                const std::int32_t nid = w.stk_ptr[ct_sp];
                const std::int32_t parent_cid = w.stk_cid_ptr[ct_sp];

                if (nid < w.row_count)
                    continue;

                const std::int32_t lc_node = w.lc_ptr[nid];
                const std::int32_t rc_node = w.rc_ptr[nid];
                if (lc_node < 0 || rc_node < 0)
                    continue;

                const std::int32_t ls = w.ns_ptr[lc_node];
                const std::int32_t rs = w.ns_ptr[rc_node];
                const Float wt = w.nw_ptr[nid];
                const Float lambda = (wt > Float(0)) ? Float(1) / wt : Float(1e30);

                const bool l_big = ls >= w.min_cluster_size;
                const bool r_big = rs >= w.min_cluster_size;

                if (l_big && r_big) {
                    const std::int32_t lcid = next_cid++;
                    const std::int32_t rcid = next_cid++;
                    w.dtc_ptr[lc_node] = lcid;
                    w.dtc_ptr[rc_node] = rcid;
                    add_condensed_edge(parent_cid, lcid, lambda, ls);
                    add_condensed_edge(parent_cid, rcid, lambda, rs);
                    w.stk_ptr[ct_sp] = lc_node;
                    w.stk_cid_ptr[ct_sp] = lcid;
                    ct_sp++;
                    w.stk_ptr[ct_sp] = rc_node;
                    w.stk_cid_ptr[ct_sp] = rcid;
                    ct_sp++;
                }
                else if (l_big) {
                    w.dtc_ptr[lc_node] = parent_cid;
                    collect_and_add_leaves(rc_node, parent_cid, lambda);
                    w.stk_ptr[ct_sp] = lc_node;
                    w.stk_cid_ptr[ct_sp] = parent_cid;
                    ct_sp++;
                }
                else if (r_big) {
                    w.dtc_ptr[rc_node] = parent_cid;
                    collect_and_add_leaves(lc_node, parent_cid, lambda);
                    w.stk_ptr[ct_sp] = rc_node;
                    w.stk_cid_ptr[ct_sp] = parent_cid;
                    ct_sp++;
                }
                else {
                    collect_and_add_leaves(lc_node, parent_cid, lambda);
                    collect_and_add_leaves(rc_node, parent_cid, lambda);
                }
            }

            w.nc_ptr[0] = next_cid;
        });
    });
}

// =========================================================================
// Cluster extraction helper: Kernel 3b
// EOM stability selection + cluster mappings + label assignment prep.
// =========================================================================
template <typename Float>
sycl::event eom_select_clusters_kernel(sycl::queue& queue,
                                       const cluster_work_ptrs<Float>& w,
                                       const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            const std::int32_t n_clusters = w.nc_ptr[0];
            if (n_clusters == 0)
                return;

            const std::int32_t root_cid = static_cast<std::int32_t>(w.row_count);
            const std::int32_t cond_count = w.cond_cnt_ptr[0];

            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                w.ilc_ptr[c] = 1;
                w.is_ptr[c] = 1;
            }
            w.csz_ptr[root_cid] = static_cast<std::int32_t>(w.row_count);

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (w.cond_c_ptr[i] >= w.row_count) {
                    w.lb_ptr[w.cond_c_ptr[i]] = w.cond_l_ptr[i];
                    w.ilc_ptr[w.cond_p_ptr[i]] = 0;
                    if (w.cc0_ptr[w.cond_p_ptr[i]] < 0)
                        w.cc0_ptr[w.cond_p_ptr[i]] = w.cond_c_ptr[i];
                    else
                        w.cc1_ptr[w.cond_p_ptr[i]] = w.cond_c_ptr[i];
                    w.csz_ptr[w.cond_c_ptr[i]] = w.cond_s_ptr[i];
                }
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                const Float birth = w.lb_ptr[w.cond_p_ptr[i]];
                const Float contrib =
                    (w.cond_l_ptr[i] - birth) * static_cast<Float>(w.cond_s_ptr[i]);
                if (contrib > Float(0))
                    w.stab_ptr[w.cond_p_ptr[i]] += contrib;
            }

            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                if (w.csz_ptr[c] < w.min_cluster_size)
                    w.is_ptr[c] = 0;
            }

            const std::int32_t mcs_max = (w.max_cluster_size > 0)
                                             ? static_cast<std::int32_t>(w.max_cluster_size)
                                             : std::numeric_limits<std::int32_t>::max();

            if (w.cluster_selection == 1) {
                for (std::int32_t c = root_cid; c < n_clusters; c++) {
                    w.is_ptr[c] = (w.ilc_ptr[c] && w.csz_ptr[c] >= w.min_cluster_size) ? 1 : 0;
                }
            }
            else {
                for (std::int32_t c = n_clusters - 1; c >= root_cid; c--) {
                    if (w.ilc_ptr[c])
                        continue;

                    Float child_sum = Float(0);
                    if (w.cc0_ptr[c] >= 0)
                        child_sum += w.stab_ptr[w.cc0_ptr[c]];
                    if (w.cc1_ptr[c] >= 0)
                        child_sum += w.stab_ptr[w.cc1_ptr[c]];

                    const Float parent_stab = (w.csz_ptr[c] > mcs_max) ? Float(0) : w.stab_ptr[c];

                    if (child_sum > parent_stab) {
                        w.is_ptr[c] = 0;
                        w.stab_ptr[c] = child_sum;
                    }
                    else {
                        std::int32_t dsp = 0;
                        if (w.cc0_ptr[c] >= 0)
                            w.stk_ptr[dsp++] = w.cc0_ptr[c];
                        if (w.cc1_ptr[c] >= 0)
                            w.stk_ptr[dsp++] = w.cc1_ptr[c];
                        while (dsp > 0) {
                            const std::int32_t d = w.stk_ptr[--dsp];
                            w.is_ptr[d] = 0;
                            if (w.cc0_ptr[d] >= 0)
                                w.stk_ptr[dsp++] = w.cc0_ptr[d];
                            if (w.cc1_ptr[d] >= 0)
                                w.stk_ptr[dsp++] = w.cc1_ptr[d];
                        }
                    }
                }
            }

            if (!w.allow_single_cluster) {
                std::int32_t sel_count = 0;
                for (std::int32_t c = root_cid; c < n_clusters; c++) {
                    if (w.is_ptr[c])
                        sel_count++;
                }
                if (sel_count == 1 && w.is_ptr[root_cid] && !w.ilc_ptr[root_cid]) {
                    w.is_ptr[root_cid] = 0;
                    if (w.cc0_ptr[root_cid] >= 0)
                        w.is_ptr[w.cc0_ptr[root_cid]] = 1;
                    if (w.cc1_ptr[root_cid] >= 0)
                        w.is_ptr[w.cc1_ptr[root_cid]] = 1;
                }
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (w.cond_c_ptr[i] >= w.row_count)
                    w.cprt_ptr[w.cond_c_ptr[i]] = w.cond_p_ptr[i];
            }

            if (w.cluster_selection_epsilon > 0.0) {
                const Float eps = static_cast<Float>(w.cluster_selection_epsilon);
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (std::int32_t c = root_cid + 1; c < n_clusters; c++) {
                        if (!w.is_ptr[c])
                            continue;
                        const Float birth_dist =
                            (w.lb_ptr[c] > Float(0)) ? Float(1) / w.lb_ptr[c] : Float(0);
                        if (birth_dist < eps) {
                            const std::int32_t parent = w.cprt_ptr[c];
                            if (parent >= root_cid && parent < n_clusters) {
                                w.is_ptr[c] = 0;
                                w.is_ptr[parent] = 1;
                                changed = true;
                            }
                        }
                    }
                }
            }

            std::int32_t label_counter = 0;
            for (std::int32_t c = root_cid; c < n_clusters; c++) {
                if (w.is_ptr[c])
                    w.clab_ptr[c] = label_counter++;
            }

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (w.cond_c_ptr[i] < w.row_count)
                    w.pff_ptr[w.cond_c_ptr[i]] = w.cond_p_ptr[i];
            }
        });
    });
}

// =========================================================================
// Cluster extraction helper: Kernels 4+5
// Build dendro_parent array, then label each point independently.
// =========================================================================
template <typename Float>
sycl::event assign_label_kernels(sycl::queue& queue,
                                 const cluster_work_ptrs<Float>& w,
                                 const bk::event_vector& deps) {
    // Kernel 4 (parallel_for): Build dendro_parent from dendrogram edges
    // Each work-item processes one dendrogram edge independently
    auto k4_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(w.edge_count), [=](sycl::id<1> idx) {
            const std::int64_t e = idx[0];
            if (w.dsz_ptr[e] > 0) {
                const std::int64_t nid = w.row_count + e;
                w.dp_ptr[w.dl_ptr[e]] = static_cast<std::int32_t>(nid);
                w.dp_ptr[w.dr_ptr[e]] = static_cast<std::int32_t>(nid);
            }
        });
    });

    // Kernel 5 (parallel_for): Label each point independently
    // Each work-item walks up cluster tree or dendrogram tree for one point
    auto k5_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k4_event });
        h.parallel_for(sycl::range<1>(w.row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t n_clusters = w.nc_ptr[0];

            // If n_clusters == 0, kernel 3 already set resp_ptr[i] = -1
            if (n_clusters == 0)
                return;

            const std::int32_t root_cid = static_cast<std::int32_t>(w.row_count);
            w.resp_ptr[i] = -1;

            // Case 1: Point fell out of a cluster — walk up to deepest selected ancestor.
            const std::int32_t fell = w.pff_ptr[i];
            if (fell >= 0) {
                std::int32_t c = fell;
                while (c >= root_cid && c < n_clusters) {
                    if (w.is_ptr[c]) {
                        w.resp_ptr[i] = w.clab_ptr[c];
                        return;
                    }
                    c = w.cprt_ptr[c];
                }
                return;
            }

            // Case 2: Point was never ejected — walk up dendrogram tree
            std::int32_t nid = static_cast<std::int32_t>(i);
            while (nid >= 0 && nid < static_cast<std::int32_t>(w.total_nodes)) {
                const std::int32_t cid = w.dtc_ptr[nid];
                if (cid >= root_cid) {
                    std::int32_t c = cid;
                    while (c >= root_cid && c < n_clusters) {
                        if (w.is_ptr[c]) {
                            w.resp_ptr[i] = w.clab_ptr[c];
                            return;
                        }
                        c = w.cprt_ptr[c];
                    }
                    return;
                }
                nid = w.dp_ptr[nid];
            }
        });
    });

    return k5_event;
}

// =========================================================================
// extract_clusters: orchestrator that allocates working memory and
// delegates to the three helper functions above.
// =========================================================================
template <typename Float>
sycl::event kernels_fp<Float>::extract_clusters(sycl::queue& queue,
                                                const pr::ndview<std::int32_t, 1>& mst_from,
                                                const pr::ndview<std::int32_t, 1>& mst_to,
                                                const pr::ndview<Float, 1>& mst_weights,
                                                pr::ndview<std::int32_t, 1>& responses,
                                                std::int64_t row_count,
                                                std::int64_t min_cluster_size,
                                                const bk::event_vector& deps,
                                                std::int32_t cluster_selection,
                                                bool allow_single_cluster,
                                                double cluster_selection_epsilon,
                                                std::int64_t max_cluster_size) {
    ONEDAL_PROFILER_TASK(hdbscan.extract_clusters, queue);

    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_cluster_size >= 2);

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t total_nodes = 2 * row_count - 1;
    const std::int64_t max_condensed = 2 * row_count;
    const std::int64_t max_clusters = total_nodes;

    // Allocate working memory on device
    auto [uf_parent, uf_parent_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_size, comp_size_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_to_node, comp_to_node_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);

    auto [dendro_left, dl_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_right, dr_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_weight, dw_ev] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [dendro_size, ds_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    auto [node_size, ns_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [left_child, lc_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);
    auto [right_child, rc_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);
    auto [node_weight, nw_ev] =
        pr::ndarray<Float, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [dendro_to_cluster, dtc_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);

    auto [cond_parent, cp_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_child, cc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_lambda, cl_ev] =
        pr::ndarray<Float, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_size, cs_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_condensed, sycl::usm::alloc::device);
    auto [cond_count_arr, cca_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    auto [stability, stab_ev] =
        pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [lambda_birth, lb_ev] =
        pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [is_leaf_cluster, ilc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [is_selected, is_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [cluster_label, clab_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);
    auto [cluster_size_arr, csz_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device);
    auto [cluster_parent, cprt_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);
    auto [child_clusters_0, cc0_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);
    auto [child_clusters_1, cc1_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, max_clusters, -1, sycl::usm::alloc::device);

    auto [point_fell_from, pff_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    auto [dendro_parent, dp_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, total_nodes, -1, sycl::usm::alloc::device);

    auto [stack_arr, stk_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [stack_cid, stk_cid_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);
    auto [leaf_stack_arr, leaf_stk_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, total_nodes, sycl::usm::alloc::device);

    auto [n_clusters_arr, nca_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    // Collect all allocation events into a single dependency vector
    bk::event_vector all_events = deps;
    all_events.insert(all_events.end(),
                      { uf_parent_ev, comp_size_ev, comp_to_node_ev, dl_ev,   dr_ev,  dw_ev,  ds_ev,
                        ns_ev,        lc_ev,        rc_ev,           nw_ev,   dtc_ev, cp_ev,  cc_ev,
                        cl_ev,        cs_ev,        cca_ev,          stab_ev, lb_ev,  ilc_ev, is_ev,
                        clab_ev,      csz_ev,       cprt_ev,         cc0_ev,  cc1_ev, pff_ev, dp_ev,
                        stk_ev,       stk_cid_ev,   leaf_stk_ev,     nca_ev });

    // Fill working data pointers
    cluster_work_ptrs<Float> w;
    w.mst_from_ptr = mst_from.get_data();
    w.mst_to_ptr = mst_to.get_data();
    w.mst_weights_ptr = mst_weights.get_data();
    w.resp_ptr = responses.get_mutable_data();
    w.uf_parent_ptr = uf_parent.get_mutable_data();
    w.comp_size_ptr = comp_size.get_mutable_data();
    w.comp_to_node_ptr = comp_to_node.get_mutable_data();
    w.dl_ptr = dendro_left.get_mutable_data();
    w.dr_ptr = dendro_right.get_mutable_data();
    w.dw_ptr = dendro_weight.get_mutable_data();
    w.dsz_ptr = dendro_size.get_mutable_data();
    w.ns_ptr = node_size.get_mutable_data();
    w.lc_ptr = left_child.get_mutable_data();
    w.rc_ptr = right_child.get_mutable_data();
    w.nw_ptr = node_weight.get_mutable_data();
    w.dtc_ptr = dendro_to_cluster.get_mutable_data();
    w.cond_p_ptr = cond_parent.get_mutable_data();
    w.cond_c_ptr = cond_child.get_mutable_data();
    w.cond_l_ptr = cond_lambda.get_mutable_data();
    w.cond_s_ptr = cond_size.get_mutable_data();
    w.cond_cnt_ptr = cond_count_arr.get_mutable_data();
    w.stab_ptr = stability.get_mutable_data();
    w.lb_ptr = lambda_birth.get_mutable_data();
    w.ilc_ptr = is_leaf_cluster.get_mutable_data();
    w.is_ptr = is_selected.get_mutable_data();
    w.clab_ptr = cluster_label.get_mutable_data();
    w.csz_ptr = cluster_size_arr.get_mutable_data();
    w.cprt_ptr = cluster_parent.get_mutable_data();
    w.cc0_ptr = child_clusters_0.get_mutable_data();
    w.cc1_ptr = child_clusters_1.get_mutable_data();
    w.pff_ptr = point_fell_from.get_mutable_data();
    w.dp_ptr = dendro_parent.get_mutable_data();
    w.stk_ptr = stack_arr.get_mutable_data();
    w.stk_cid_ptr = stack_cid.get_mutable_data();
    w.leaf_stk_ptr = leaf_stack_arr.get_mutable_data();
    w.nc_ptr = n_clusters_arr.get_mutable_data();
    w.row_count = row_count;
    w.edge_count = edge_count;
    w.min_cluster_size = min_cluster_size;
    w.total_nodes = total_nodes;
    w.cluster_selection = cluster_selection;
    w.allow_single_cluster = allow_single_cluster;
    w.cluster_selection_epsilon = cluster_selection_epsilon;
    w.max_cluster_size = max_cluster_size;

    // Submit kernels via helpers
    auto k2_event = build_dendrogram_kernels(queue, w, all_events);
    auto k3a_event = build_condensed_tree_kernel(queue, w, { k2_event });
    auto k3b_event = eom_select_clusters_kernel(queue, w, { k3a_event });
    return assign_label_kernels(queue, w, { k3b_event });
}

#endif

} // namespace oneapi::dal::hdbscan::backend
