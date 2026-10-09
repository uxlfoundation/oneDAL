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

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/backend/atomic.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/search/copy_search_callback.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/// Device pointers and scalar parameters shared by the GPU cluster-extraction kernels.
///
/// @tparam Float Floating-point type used for MST weights and lambdas
template <typename Float>
struct cluster_work_ptrs {
    const std::int32_t* mst_from_ptr;
    const std::int32_t* mst_to_ptr;
    const Float* mst_weights_ptr;
    std::int32_t* resp_ptr;

    std::int32_t* uf_parent_ptr;
    std::int32_t* comp_size_ptr;
    std::int32_t* comp_to_node_ptr;

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

    // Membership probabilities and their scratch; all null unless probabilities are requested.
    Float* prob_ptr;
    Float* cdeath_ptr;
    Float* ldeath_ptr;
    Float* plam_ptr;

    // Single-linkage tree output, `(row_count - 1) x 4`; null unless requested.
    Float* slt_ptr;

    std::int64_t row_count;
    std::int64_t edge_count;
    std::int64_t min_cluster_size;
    std::int64_t total_nodes;
    std::int32_t cluster_selection; // 0 = EOM, 1 = leaf
    bool allow_single_cluster;
    // `Float`, not `double`, so kernels capturing this struct need no fp64 support.
    Float cluster_selection_epsilon;
    std::int64_t max_cluster_size;
};

/// The brute-force path may use at most `1 / mrd_global_mem_divisor` of device global memory.
///
/// An oversubscribed `malloc_device` succeeds and then faults on first access instead of raising.
constexpr std::int64_t mrd_global_mem_divisor = 2;

/// Byte total the sizing helpers return on overflow; larger than any device limit.
constexpr std::int64_t mrd_bytes_saturated = dal::detail::limits<std::int64_t>::max();

/// Multiply two non-negative byte counts without signed overflow.
///
/// @param[in] lhs First factor
/// @param[in] rhs Second factor
///
/// @return `lhs * rhs`, or `mrd_bytes_saturated` when that product would not fit
inline std::int64_t mrd_mul_bytes(std::int64_t lhs, std::int64_t rhs) {
    ONEDAL_ASSERT(lhs >= 0);
    ONEDAL_ASSERT(rhs >= 0);
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    return (lhs > mrd_bytes_saturated / rhs) ? mrd_bytes_saturated : lhs * rhs;
}

/// Add two non-negative byte counts without signed overflow.
///
/// @param[in] lhs First summand
/// @param[in] rhs Second summand
///
/// @return `lhs + rhs`, or `mrd_bytes_saturated` when that sum would not fit
inline std::int64_t mrd_add_bytes(std::int64_t lhs, std::int64_t rhs) {
    ONEDAL_ASSERT(lhs >= 0);
    ONEDAL_ASSERT(rhs >= 0);
    return (lhs > mrd_bytes_saturated - rhs) ? mrd_bytes_saturated : lhs + rhs;
}

/// Decide whether the brute-force path's device buffers fit within the given limits.
///
/// The peak is the matrix plus the larger of the k-selection buffers and the MST edge arrays,
/// which are never live at the same time.
///
/// @param[in] row_count             Number of input rows `n`
/// @param[in] column_count          Number of input columns `d`
/// @param[in] element_size          Size in bytes of one distance-matrix element
/// @param[in] max_alloc_bytes       Device limit on a single allocation
/// @param[in] global_mem_bytes      Device global memory size
/// @param[in] min_samples           `k` of the core-distance k-selection
/// @param[in] kselect_scratch_bytes Device scratch of `kselect_by_rows`, from
///                                  `pr::kselect_by_rows_scratch_size`
///
/// @return `true` if the `n x n` matrix fits one allocation and the peak footprint fits the
///         global-memory budget
inline bool mrd_matrix_fits_on_device(std::int64_t row_count,
                                      std::int64_t column_count,
                                      std::int64_t element_size,
                                      std::int64_t max_alloc_bytes,
                                      std::int64_t global_mem_bytes,
                                      std::int64_t min_samples,
                                      std::int64_t kselect_scratch_bytes) {
    const std::int64_t row_bytes = mrd_mul_bytes(element_size, row_count);
    const std::int64_t matrix_bytes = mrd_mul_bytes(row_bytes, row_count);
    if (matrix_bytes > max_alloc_bytes) {
        return false;
    }

    // The input table and the core distances.
    const std::int64_t resident_bytes = mrd_mul_bytes(row_bytes, column_count + 1);

    // The `n x min_samples` k-selection output and its scratch.
    const std::int64_t selection_bytes =
        mrd_add_bytes(mrd_mul_bytes(row_bytes, min_samples), kselect_scratch_bytes);

    // The three MST edge arrays.
    const std::int64_t mst_bytes =
        mrd_mul_bytes(2 * static_cast<std::int64_t>(sizeof(std::int32_t)) + element_size,
                      row_count);

    const std::int64_t peak_bytes = mrd_add_bytes(mrd_add_bytes(matrix_bytes, resident_bytes),
                                                  std::max(selection_bytes, mst_bytes));
    return peak_bytes <= global_mem_bytes / mrd_global_mem_divisor;
}

/// Reject the row counts whose `n x n` MRD matrix does not fit on the device.
///
/// @tparam Float Floating-point type of the distance matrix
///
/// @param[in] queue        The SYCL queue whose device has to hold the matrix
/// @param[in] row_count    Number of input rows `n`
/// @param[in] column_count Number of input columns `d`
/// @param[in] min_samples  `k` of the core-distance k-selection
///
/// @throws domain_error if the matrix exceeds the single-allocation limit or if
///         the pipeline's peak footprint exceeds device global memory
template <typename Float>
inline void check_mrd_matrix_fits_on_device(sycl::queue& queue,
                                            std::int64_t row_count,
                                            std::int64_t column_count,
                                            std::int64_t min_samples) {
    constexpr std::int64_t element_size = static_cast<std::int64_t>(sizeof(Float));
    const std::int64_t max_alloc_bytes = bk::device_max_mem_alloc_size(queue);
    const std::int64_t global_mem_bytes = bk::device_global_mem_size(queue);

    // Size the matrix alone first: the scratch query needs an `n x n` shape to be representable.
    const bool matrix_fits = mrd_matrix_fits_on_device(row_count,
                                                       column_count,
                                                       element_size,
                                                       max_alloc_bytes,
                                                       global_mem_bytes,
                                                       min_samples,
                                                       0);
    const std::int64_t kselect_scratch_bytes =
        matrix_fits
            ? pr::kselect_by_rows_scratch_size<Float>(queue, { row_count, row_count }, min_samples)
            : 0;
    if (!matrix_fits || !mrd_matrix_fits_on_device(row_count,
                                                   column_count,
                                                   element_size,
                                                   max_alloc_bytes,
                                                   global_mem_bytes,
                                                   min_samples,
                                                   kselect_scratch_bytes)) {
        throw domain_error(
            dal::detail::error_messages::hdbscan_brute_force_matrix_does_not_fit_on_device());
    }
}

/// Match scikit-learn on zero-norm rows of a square cosine distance matrix.
///
/// The cosine primitive divides by the row norms, so a zero row yields NaN. scikit-learn
/// normalizes it to the zero vector instead, which puts it at distance 1 from every other row.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue The SYCL queue
/// @param[in]     data  Input data of size `n x d`
/// @param[in,out] dist  Cosine distance matrix of size `n x n`
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event fix_zero_norm_cosine(sycl::queue& queue,
                                        const pr::ndview<Float, 2>& data,
                                        pr::ndview<Float, 2>& dist,
                                        const bk::event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    pr::ndarray<Float, 1> norms;
    sycl::event norms_event;
    std::tie(norms, norms_event) = pr::compute_squared_l2_norms(queue, data, deps);
    const Float* const norms_ptr = norms.get_data();
    Float* const dist_ptr = dist.get_mutable_data();
    const std::int64_t stride = dist.get_leading_stride();

    auto event = bk::parallel_for_2d_by_row_blocks(
        queue,
        n,
        n,
        [=](std::int64_t i, std::int64_t j) {
            if (norms_ptr[i] == Float(0) || norms_ptr[j] == Float(0)) {
                dist_ptr[i * stride + j] = (i == j) ? Float(0) : Float(1);
            }
        },
        { norms_event });
    // `norms` is freed on return and `sycl::free` does not wait.
    event.wait_and_throw();
    return event;
}

/// Compute the full pairwise distance matrix; squared L2 for `euclidean`, final distances otherwise.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue  The SYCL queue
/// @param[in]  data   Input matrix of size `n x d`
/// @param[out] dist   Output matrix of size `n x n` (row-major)
/// @param[in]  metric Distance metric tag (`distance_metric`)
/// @param[in]  degree Minkowski degree (used only when `metric == minkowski`)
/// @param[in]  deps   Events that must complete before submission
///
/// @return Event signaling completion of the distance computation
template <typename Float>
inline sycl::event compute_distance_matrix(sycl::queue& queue,
                                           const pr::ndview<Float, 2>& data,
                                           pr::ndview<Float, 2>& dist,
                                           distance_metric metric,
                                           double degree,
                                           const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_distance_matrix, queue);

    const std::int64_t n = data.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);

    switch (metric) {
        case distance_metric::cosine: {
            pr::cosine_distance<Float> dist_op(queue);
            const auto dist_event = dist_op(data, data, dist, deps);
            return fix_zero_norm_cosine<Float>(queue, data, dist, { dist_event });
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
        default: { // euclidean: squared L2, the root is taken later
            pr::squared_l2_distance<Float> dist_op(queue);
            return dist_op(data, data, dist, deps);
        }
    }
}

/// Compute per-point core distances as the `min_samples`-th smallest entry of each row.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  dist           Pairwise distance matrix of size `n x n`
/// @param[out] core_distances Per-point core distances, length `n`
/// @param[in]  min_samples    `k` used for the k-NN core-distance definition
/// @param[in]  row_count      Number of rows `n`
/// @param[in]  metric         Distance metric tag (controls the sqrt finalize)
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event compute_core_distances(sycl::queue& queue,
                                          const pr::ndview<Float, 2>& dist,
                                          pr::ndview<Float, 1>& core_distances,
                                          std::int64_t min_samples,
                                          std::int64_t row_count,
                                          distance_metric metric,
                                          const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances, queue);

    const std::int64_t n = row_count;

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(dist.get_dimension(0) == n);
    ONEDAL_ASSERT(dist.get_dimension(1) == n);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == n);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= n);

    // The point itself counts as its first neighbor, as in scikit-learn.
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

    // `ksel_vals` is freed on return and `sycl::free` does not wait for its readers.
    extract_event.wait_and_throw();
    return extract_event;
}

/// Compute per-point core distances with `pr::search_engine`, without the `n x n` matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  data           Input data of size `n x d`, device USM
/// @param[out] core_distances Per-point core distances, length `n`
/// @param[in]  min_samples    `k` used for the k-NN core-distance definition
/// @param[in]  block_size     Query rows per search block, in `[1, n]`
/// @param[in]  metric         Distance metric tag; cosine is not supported
/// @param[in]  degree         Minkowski degree
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion; already complete on return
template <typename Float>
inline sycl::event compute_core_distances_blocked(sycl::queue& queue,
                                                  const pr::ndview<Float, 2>& data,
                                                  pr::ndview<Float, 1>& core_distances,
                                                  std::int64_t min_samples,
                                                  std::int64_t block_size,
                                                  distance_metric metric,
                                                  double degree,
                                                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances_blocked, queue);

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t k = min_samples;
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(block_size >= 1 && block_size <= n);
    ONEDAL_ASSERT(k >= 1 && k <= n);
    ONEDAL_ASSERT(metric != distance_metric::cosine);

    auto knn_distances = pr::ndarray<Float, 2>::empty(queue, { n, k }, sycl::usm::alloc::device);
    pr::copy_callback<Float, false, true> callback(queue, block_size, {}, knn_distances);
    const std::int64_t train_block = pr::propose_train_block<Float>(queue, data.get_dimension(1));

    sycl::event search_event;
    const bool is_euclidean = (metric == distance_metric::euclidean);
    if (is_euclidean) {
        const pr::search_engine<Float, pr::squared_l2_distance<Float>> search(queue,
                                                                              data,
                                                                              train_block);
        search_event = search(data, callback, block_size, k, deps);
    }
    else if (metric == distance_metric::chebyshev) {
        const pr::chebyshev_distance<Float> dist(queue);
        const pr::search_engine<Float, pr::chebyshev_distance<Float>> search(queue,
                                                                             data,
                                                                             train_block,
                                                                             dist);
        search_event = search(data, callback, block_size, k, deps);
    }
    else {
        const Float p =
            metric == distance_metric::manhattan ? Float(1) : static_cast<Float>(degree);
        const pr::lp_distance<Float> dist(queue, pr::lp_metric<Float>(p));
        const pr::search_engine<Float, pr::lp_distance<Float>> search(queue,
                                                                      data,
                                                                      train_block,
                                                                      dist);
        search_event = search(data, callback, block_size, k, deps);
    }

    const Float* const knn_ptr = knn_distances.get_data();
    Float* const core_ptr = core_distances.get_mutable_data();
    auto extract_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(search_event);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float val = knn_ptr[i * k + (k - 1)];
            core_ptr[i] = is_euclidean ? sycl::sqrt(sycl::fmax(val, Float(0))) : val;
        });
    });

    // `knn_distances` is freed on return and `sycl::free` does not wait for its readers.
    extract_event.wait_and_throw();
    return extract_event;
}

/// Convert a distance matrix in place into `MRD(i, j) = max(core_i, core_j, dist(i, j)) / alpha`.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue          The SYCL queue
/// @param[in]     core_distances Per-point core distances, length `n` (unscaled)
/// @param[in,out] mrd_matrix     Distance matrix `n x n` (squared L2 for euclidean), overwritten
///                               with MRD values
/// @param[in]     metric         Distance metric tag (controls the sqrt finalize)
/// @param[in]     alpha          Robust single-linkage scaling factor; divides the whole MRD
/// @param[in]     deps           Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event compute_mrd_matrix(sycl::queue& queue,
                                      const pr::ndview<Float, 1>& core_distances,
                                      pr::ndview<Float, 2>& mrd_matrix,
                                      distance_metric metric,
                                      double alpha,
                                      const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_mrd_matrix, queue);

    const std::int64_t n = core_distances.get_dimension(0);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(0) == n);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    const Float* core_ptr = core_distances.get_data();
    Float* mrd_ptr = mrd_matrix.get_mutable_data();
    const bool needs_sqrt = (metric == distance_metric::euclidean);
    const Float inv_alpha = static_cast<Float>(1.0 / alpha);

    return bk::parallel_for_2d_by_row_blocks(
        queue,
        n,
        n,
        [=](std::int64_t i, std::int64_t j) {
            // Euclidean entries hold squared L2; the other metrics hold final distances.
            const Float d = needs_sqrt ? sycl::sqrt(sycl::fmax(mrd_ptr[i * n + j], Float(0)))
                                       : mrd_ptr[i * n + j];
            // scikit-learn's brute path scales the core distances by 1/alpha as well.
            const Float cd_i = core_ptr[i] * inv_alpha;
            const Float cd_j = core_ptr[j] * inv_alpha;
            mrd_ptr[i * n + j] = sycl::fmax(sycl::fmax(cd_i, cd_j), d * inv_alpha);
        },
        deps);
}

/// Find each point's nearest neighbor in another component from a precomputed MRD matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue           The SYCL queue
/// @param[in]  mrd_ptr         Precomputed MRD matrix of size `n x n`
/// @param[in]  comp_ptr        Per-point component id, length `n`
/// @param[out] pt_best_mrd_ptr Per-point best MRD, length `n`
/// @param[out] pt_best_idx_ptr Per-point best different-component column index, length `n`
/// @param[in]  n               Number of points
/// @param[in]  deps            Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_find_nearest_mrd(sycl::queue& queue,
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

/// Find each point's nearest neighbor in another component, computing the distances on the fly.
///
/// Uses `MRD(i, j) = max(core_i, core_j, dist(i, j) * inv_alpha)`, so no `n x n` matrix is held.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue           The SYCL queue
/// @param[in]  data_ptr        Row-major input buffer, size `n x col_count`
/// @param[in]  col_count       Number of features
/// @param[in]  core_ptr        Per-point core distances, length `n` (unscaled)
/// @param[in]  comp_ptr        Per-point component id, length `n`
/// @param[out] pt_best_mrd_ptr Per-point best MRD, length `n`
/// @param[out] pt_best_idx_ptr Per-point best different-component point index, length `n`
/// @param[in]  n               Number of points
/// @param[in]  metric_id       0=euclidean, 1=manhattan, 2=minkowski, 3=chebyshev
/// @param[in]  degree          Minkowski degree (used only when `metric_id == 2`)
/// @param[in]  inv_alpha       `1.0 / alpha`, applied only to dist(i,j) inside MRD
/// @param[in]  deps            Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_find_nearest_otf(sycl::queue& queue,
                                            const Float* data_ptr,
                                            std::int64_t col_count,
                                            const Float* core_ptr,
                                            const std::int32_t* comp_ptr,
                                            Float* pt_best_mrd_ptr,
                                            std::int32_t* pt_best_idx_ptr,
                                            std::int64_t n,
                                            std::int32_t metric_id,
                                            Float degree,
                                            Float inv_alpha,
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
                Float mrd = sycl::fmax(sycl::fmax(my_core, core_ptr[j]), dist * inv_alpha);
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

/// Reduce per-point bests to per-component bests, then merge components and append the MST edges.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue              The SYCL queue
/// @param[in,out] comp_ptr           Per-point component id, length `n` (re-read after compress)
/// @param[in,out] uf_parent_ptr      Union-find parent array, length `n`
/// @param[in,out] uf_rank_ptr        Union-find rank array, length `n`
/// @param[in]     pt_best_mrd_ptr    Per-point best MRD from the find phase
/// @param[in]     pt_best_idx_ptr    Per-point best different-component index from the find phase
/// @param[out]    comp_best_mrd_ptr  Scratch: per-component best MRD, length `n`
/// @param[out]    comp_best_from_ptr Scratch: per-component best `from` index, length `n`
/// @param[out]    mst_from_ptr       Output MST `from` endpoints
/// @param[out]    mst_to_ptr         Output MST `to` endpoints
/// @param[out]    mst_weight_ptr     Output MST weights
/// @param[in,out] edges_added_ptr    Single-element counter of MST edges added so far
/// @param[in,out] num_comp_ptr       Single-element counter of remaining components
/// @param[in]     n                  Number of points
/// @param[in]     deps               Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event boruvka_merge_components(sycl::queue& queue,
                                            std::int32_t* comp_ptr,
                                            std::int32_t* uf_parent_ptr,
                                            std::int32_t* uf_rank_ptr,
                                            const Float* pt_best_mrd_ptr,
                                            const std::int32_t* pt_best_idx_ptr,
                                            Float* comp_best_mrd_ptr,
                                            std::int32_t* comp_best_from_ptr,
                                            std::int32_t* mst_from_ptr,
                                            std::int32_t* mst_to_ptr,
                                            Float* mst_weight_ptr,
                                            std::int32_t* edges_added_ptr,
                                            std::int32_t* num_comp_ptr,
                                            std::int64_t n,
                                            const bk::event_vector& deps) {
    const Float inf = std::numeric_limits<Float>::max();
    constexpr std::int32_t no_point = std::numeric_limits<std::int32_t>::max();
    const auto range = sycl::range<1>(n);

    auto reset_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> c) {
            comp_best_mrd_ptr[c] = inf;
            comp_best_from_ptr[c] = no_point;
        });
    });

    // Two atomic passes reproduce the serial reduction exactly: the smallest MRD per component,
    // then the smallest point index among the points that reach it.
    auto min_mrd_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(reset_event);
        h.parallel_for(range, [=](sycl::id<1> i) {
            if (pt_best_idx_ptr[i] >= 0) {
                bk::atomic_global_min(comp_best_mrd_ptr + comp_ptr[i], pt_best_mrd_ptr[i]);
            }
        });
    });

    auto min_idx_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(min_mrd_event);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            const std::int32_t i = static_cast<std::int32_t>(idx[0]);
            const std::int32_t c = comp_ptr[i];
            const Float mrd = pt_best_mrd_ptr[i];
            if (pt_best_idx_ptr[i] >= 0 && mrd < inf && mrd == comp_best_mrd_ptr[c]) {
                bk::atomic_global_min(comp_best_from_ptr + c, i);
            }
        });
    });

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(min_idx_event);
        h.single_task([=]() {
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
                const std::int32_t u = comp_best_from_ptr[c];
                if (u == no_point)
                    continue;
                const std::int32_t v = pt_best_idx_ptr[u];
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

/// Set `comp[i] = find(i)` for every point after a Boruvka merge round.
///
/// @param[in]  queue         The SYCL queue
/// @param[out] comp_ptr      Per-point component id, length `n` (overwritten)
/// @param[in]  uf_parent_ptr Union-find parent array, length `n`
/// @param[in]  n             Number of points
/// @param[in]  deps          Events that must complete before submission
///
/// @return Event signaling completion
inline sycl::event boruvka_compress_components(sycl::queue& queue,
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

/// Build the MST under MRD with Boruvka's algorithm from a precomputed MRD matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue       The SYCL queue
/// @param[in]  mrd_matrix  Precomputed MRD matrix of size `n x n`
/// @param[out] mst_from    Output MST `from` endpoints, length `n - 1`
/// @param[out] mst_to      Output MST `to` endpoints, length `n - 1`
/// @param[out] mst_weights Output MST weights, length `n - 1`
/// @param[in]  row_count   Number of points `n`
/// @param[in]  deps        Events that must complete before submission
///
/// @return Event signaling completion of the final Boruvka round
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline sycl::event build_mst(sycl::queue& queue,
                             const pr::ndview<Float, 2>& mrd_matrix,
                             pr::ndview<std::int32_t, 1>& mst_from,
                             pr::ndview<std::int32_t, 1>& mst_to,
                             pr::ndview<Float, 1>& mst_weights,
                             std::int64_t row_count,
                             const bk::event_vector& deps = {}) {
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
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(init_deps.end(),
                     { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, ea_ev, nc_ev });

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
    init_event.wait_and_throw();

    bool connected = (n <= 1);
    for (std::int32_t round = 0; round < max_rounds; round++) {
        // Step A: nearest neighbor in another component
        auto find_event = boruvka_find_nearest_mrd<Float>(queue,
                                                          mrd_ptr,
                                                          comp_ptr,
                                                          pt_best_mrd.get_mutable_data(),
                                                          pt_best_idx.get_mutable_data(),
                                                          n,
                                                          { last_event });
        find_event.wait_and_throw();

        // Step B: reduce and merge
        auto merge_event = boruvka_merge_components<Float>(queue,
                                                           comp_ptr,
                                                           uf_parent_ptr,
                                                           uf_rank_ptr,
                                                           pt_best_mrd.get_data(),
                                                           pt_best_idx.get_data(),
                                                           comp_best_mrd.get_mutable_data(),
                                                           comp_best_from.get_mutable_data(),
                                                           mst_from.get_mutable_data(),
                                                           mst_to.get_mutable_data(),
                                                           mst_weights.get_mutable_data(),
                                                           edges_added_arr.get_mutable_data(),
                                                           num_comp_arr.get_mutable_data(),
                                                           n,
                                                           { find_event });
        merge_event.wait_and_throw();

        // Step C: path compression
        auto compress_event =
            boruvka_compress_components(queue, comp_ptr, uf_parent_ptr, n, { merge_event });
        compress_event.wait_and_throw();

        // Check termination
        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        const std::int32_t nc = num_comp_host.get_data()[0];
        last_event = compress_event;
        if (nc <= 1) {
            connected = true;
            break;
        }
    }

    // Only non-finite input leaves components that no round can join.
    if (!connected) {
        throw domain_error(dal::detail::error_messages::hdbscan_input_data_is_not_finite());
    }
    return last_event;
}

/// Build the MST under MRD with Boruvka's algorithm, computing the distances on the fly.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  data           Row-major input buffer of size `n x col_count`
/// @param[in]  core_distances Per-point core distances, length `n`
/// @param[out] mst_from       Output MST `from` endpoints, length `n - 1`
/// @param[out] mst_to         Output MST `to` endpoints, length `n - 1`
/// @param[out] mst_weights    Output MST weights, length `n - 1`
/// @param[in]  row_count      Number of points `n`
/// @param[in]  col_count      Number of features `d`
/// @param[in]  metric         Distance metric tag
/// @param[in]  degree         Minkowski degree (used only when `metric == minkowski`)
/// @param[in]  alpha          Robust single-linkage scaling factor of `dist(i, j)` inside MRD
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion of the final Boruvka round
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline sycl::event build_mst_otf(sycl::queue& queue,
                                 const pr::ndview<Float, 2>& data,
                                 const pr::ndview<Float, 1>& core_distances,
                                 pr::ndview<std::int32_t, 1>& mst_from,
                                 pr::ndview<std::int32_t, 1>& mst_to,
                                 pr::ndview<Float, 1>& mst_weights,
                                 std::int64_t row_count,
                                 std::int64_t col_count,
                                 distance_metric metric,
                                 double degree,
                                 double alpha,
                                 const bk::event_vector& deps = {}) {
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
    auto [edges_added_arr, ea_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);
    auto [num_comp_arr, nc_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, 1, sycl::usm::alloc::device);

    std::int32_t* comp_ptr = comp.get_mutable_data();
    std::int32_t* uf_parent_ptr = uf_parent.get_mutable_data();
    std::int32_t* uf_rank_ptr = uf_rank.get_mutable_data();
    std::int32_t* num_comp_ptr = num_comp_arr.get_mutable_data();

    bk::event_vector init_deps = deps;
    init_deps.insert(init_deps.end(),
                     { comp_ev, uf_ev, ur_ev, pbm_ev, pbi_ev, cbm_ev, cbf_ev, ea_ev, nc_ev });

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
    const Float inv_alpha = static_cast<Float>(1.0 / alpha);
    const std::int32_t max_rounds = 64;
    sycl::event last_event = init_event;
    init_event.wait_and_throw();

    bool connected = (n <= 1);
    for (std::int32_t round = 0; round < max_rounds; round++) {
        auto find_event = boruvka_find_nearest_otf<Float>(queue,
                                                          data_ptr,
                                                          col_count,
                                                          core_ptr,
                                                          comp_ptr,
                                                          pt_best_mrd.get_mutable_data(),
                                                          pt_best_idx.get_mutable_data(),
                                                          n,
                                                          metric_id,
                                                          deg_f,
                                                          inv_alpha,
                                                          { last_event });
        find_event.wait_and_throw();

        auto merge_event = boruvka_merge_components<Float>(queue,
                                                           comp_ptr,
                                                           uf_parent_ptr,
                                                           uf_rank_ptr,
                                                           pt_best_mrd.get_data(),
                                                           pt_best_idx.get_data(),
                                                           comp_best_mrd.get_mutable_data(),
                                                           comp_best_from.get_mutable_data(),
                                                           mst_from.get_mutable_data(),
                                                           mst_to.get_mutable_data(),
                                                           mst_weights.get_mutable_data(),
                                                           edges_added_arr.get_mutable_data(),
                                                           num_comp_arr.get_mutable_data(),
                                                           n,
                                                           { find_event });
        merge_event.wait_and_throw();

        auto compress_event =
            boruvka_compress_components(queue, comp_ptr, uf_parent_ptr, n, { merge_event });
        compress_event.wait_and_throw();

        auto num_comp_host = num_comp_arr.to_host(queue, { compress_event });
        const std::int32_t nc = num_comp_host.get_data()[0];
        last_event = compress_event;
        if (nc <= 1) {
            connected = true;
            break;
        }
    }

    // Only non-finite input leaves components that no round can join.
    if (!connected) {
        throw domain_error(dal::detail::error_messages::hdbscan_input_data_is_not_finite());
    }
    return last_event;
}

/// Sort MST edges in ascending order of weight, keeping the endpoints aligned.
///
/// @tparam Float Floating-point type used for edge weights
///
/// @param[in]     queue       The SYCL queue
/// @param[in,out] mst_from    Source endpoints, length `edge_count` (sorted in place)
/// @param[in,out] mst_to      Target endpoints, length `edge_count` (sorted in place)
/// @param[in,out] mst_weights Edge weights (sort key), length `edge_count` (sorted in place)
/// @param[in]     edge_count  Number of MST edges
/// @param[in]     deps        Events that must complete before submission
///
/// @return Event signaling completion of the final copy-back
template <typename Float>
inline sycl::event sort_mst_by_weight(sycl::queue& queue,
                                      pr::ndview<std::int32_t, 1>& mst_from,
                                      pr::ndview<std::int32_t, 1>& mst_to,
                                      pr::ndview<Float, 1>& mst_weights,
                                      std::int64_t edge_count,
                                      const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.sort_mst_by_weight, queue);

    ONEDAL_ASSERT(edge_count > 0);

    using Index = std::uint32_t;
    auto indices = pr::ndarray<Index, 1>::empty(queue, edge_count, sycl::usm::alloc::device);
    auto iota_event = indices.arange(queue, deps);

    pr::radix_sort_indices_inplace<Float, Index> sorter(queue);
    auto sort_event = sorter(mst_weights, indices, { iota_event });
    sort_event.wait_and_throw();

    auto [sorted_from, sf_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [sorted_to, st_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    sf_event.wait_and_throw();
    st_event.wait_and_throw();

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
    permute_event.wait_and_throw();

    const std::size_t bytes = edge_count * sizeof(std::int32_t);
    auto copy_from_event = queue.memcpy(mst_from.get_mutable_data(), sf_ptr, bytes, permute_event);
    auto copy_to_event = queue.memcpy(mst_to.get_mutable_data(), st_ptr, bytes, permute_event);
    // `sorted_from`, `sorted_to` and `indices` are freed on return.
    copy_from_event.wait_and_throw();
    copy_to_event.wait_and_throw();

    return copy_to_event;
}

/// Cluster extraction kernel 1: build the single-linkage dendrogram from the sorted MST edges.
///
/// Ids `[0, row_count)` are leaves; internal node `row_count + e` is created by edge `e`.
///
/// @tparam Float Floating-point type used for edge weights
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`)
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event build_dendrogram_kernels(sycl::queue& queue,
                                            const cluster_work_ptrs<Float>& w,
                                            const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            for (std::int64_t i = 0; i < w.row_count; i++) {
                w.uf_parent_ptr[i] = static_cast<std::int32_t>(i);
                w.comp_size_ptr[i] = 1;
                w.comp_to_node_ptr[i] = static_cast<std::int32_t>(i);
                w.ns_ptr[i] = 1;
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

                const std::int32_t nid = static_cast<std::int32_t>(w.row_count + e);
                const std::int32_t new_size = w.comp_size_ptr[ru] + w.comp_size_ptr[rv];

                w.lc_ptr[nid] = w.comp_to_node_ptr[ru];
                w.rc_ptr[nid] = w.comp_to_node_ptr[rv];
                w.nw_ptr[nid] = w.mst_weights_ptr[e];
                w.ns_ptr[nid] = new_size;

                if (w.comp_size_ptr[ru] < w.comp_size_ptr[rv]) {
                    w.uf_parent_ptr[ru] = rv;
                    w.comp_size_ptr[rv] = new_size;
                    w.comp_to_node_ptr[rv] = nid;
                }
                else {
                    w.uf_parent_ptr[rv] = ru;
                    w.comp_size_ptr[ru] = new_size;
                    w.comp_to_node_ptr[ru] = nid;
                }
            }
        });
    });
}

/// Write the single-linkage dendrogram as a row-major `(row_count - 1) x 4` matrix.
///
/// Row `e` is the merge that created node `row_count + e`: `[left, right, distance, size]`, the
/// layout of scipy's `linkage` and scikit-learn's `_single_linkage_tree_`.
///
/// @tparam Float Floating-point type used for merge distances
///
/// @param[in] queue The SYCL queue
/// @param[in] w     Working pointers and constants (see `cluster_work_ptrs`);
///                  `slt_ptr` must point at a `4 * edge_count`-long buffer
/// @param[in] deps  Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event dump_single_linkage_tree_kernel(sycl::queue& queue,
                                                   const cluster_work_ptrs<Float>& w,
                                                   const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(w.edge_count), [=](sycl::id<1> idx) {
            const std::int64_t e = idx[0];
            const std::int64_t nid = w.row_count + e;
            Float* r = w.slt_ptr + 4 * e;
            r[0] = static_cast<Float>(w.lc_ptr[nid]);
            r[1] = static_cast<Float>(w.rc_ptr[nid]);
            r[2] = w.nw_ptr[nid];
            r[3] = static_cast<Float>(w.ns_ptr[nid]);
        });
    });
}

/// Cluster extraction kernel 2: build the condensed tree from the dendrogram.
///
/// A side smaller than `min_cluster_size` falls out as points; a new cluster id is allocated only
/// when both sides of a split survive.
///
/// @tparam Float Floating-point type used for edge weights / lambdas
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`)
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event build_condensed_tree_kernel(sycl::queue& queue,
                                               const cluster_work_ptrs<Float>& w,
                                               const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            std::int32_t root = -1;
            for (std::int64_t e = w.edge_count - 1; e >= 0; e--) {
                const std::int64_t nid = w.row_count + e;
                if (w.ns_ptr[nid] > 0) {
                    root = static_cast<std::int32_t>(nid);
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

            // Breadth-first, left before right, so the cluster ids match scikit-learn's.
            std::int32_t ct_head = 0;
            std::int32_t ct_sp = 0;
            w.stk_ptr[ct_sp] = root;
            w.stk_cid_ptr[ct_sp] = w.dtc_ptr[root];
            ct_sp++;

            while (ct_head < ct_sp) {
                const std::int32_t nid = w.stk_ptr[ct_head];
                const std::int32_t parent_cid = w.stk_cid_ptr[ct_head];
                ct_head++;

                if (nid < w.row_count)
                    continue;

                const std::int32_t lc_node = w.lc_ptr[nid];
                const std::int32_t rc_node = w.rc_ptr[nid];
                if (lc_node < 0 || rc_node < 0)
                    continue;

                const std::int32_t ls = w.ns_ptr[lc_node];
                const std::int32_t rs = w.ns_ptr[rc_node];
                const Float wt = w.nw_ptr[nid];
                // A zero-distance merge gets the largest finite lambda, as on the CPU.
                const Float lambda =
                    (wt > Float(0)) ? Float(1) / wt : dal::detail::limits<Float>::max();

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

/// Cluster extraction kernel 3: select clusters (EOM or leaf, then epsilon) and number them.
///
/// @tparam Float Floating-point type used for cluster lambdas / stabilities
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`)
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event eom_select_clusters_kernel(sycl::queue& queue,
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
            // The root can be selected only when `allow_single_cluster` is set.
            if (!w.allow_single_cluster)
                w.is_ptr[root_cid] = 0;
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

            // Kept 64-bit: narrowing a large cap would make it negative.
            const std::int64_t mcs_max = (w.max_cluster_size > 0)
                                             ? w.max_cluster_size
                                             : std::numeric_limits<std::int64_t>::max();

            if (w.cluster_selection == 1) {
                // Leaf mode never selects the root; `allow_single_cluster` only relaxes the
                // labeling threshold there.
                w.is_ptr[root_cid] = 0;
                for (std::int32_t c = root_cid + 1; c < n_clusters; c++) {
                    w.is_ptr[c] = (w.ilc_ptr[c] && w.csz_ptr[c] >= w.min_cluster_size) ? 1 : 0;
                }
            }
            else {
                const std::int32_t tree_top = w.allow_single_cluster ? root_cid : (root_cid + 1);
                if (w.allow_single_cluster) {
                    // scikit-learn caps the root by the points its child clusters hold.
                    std::int32_t root_size = 0;
                    if (w.cc0_ptr[root_cid] >= 0)
                        root_size += w.csz_ptr[w.cc0_ptr[root_cid]];
                    if (w.cc1_ptr[root_cid] >= 0)
                        root_size += w.csz_ptr[w.cc1_ptr[root_cid]];
                    w.csz_ptr[root_cid] = root_size;
                }
                for (std::int32_t c = n_clusters - 1; c >= tree_top; c--) {
                    if (w.ilc_ptr[c]) {
                        // Only the size cap can unselect a leaf.
                        if (w.csz_ptr[c] > mcs_max) {
                            w.is_ptr[c] = 0;
                            w.stab_ptr[c] = Float(0);
                        }
                        continue;
                    }

                    Float child_sum = Float(0);
                    if (w.cc0_ptr[c] >= 0)
                        child_sum += w.stab_ptr[w.cc0_ptr[c]];
                    if (w.cc1_ptr[c] >= 0)
                        child_sum += w.stab_ptr[w.cc1_ptr[c]];

                    const bool oversized = (w.csz_ptr[c] > mcs_max);

                    if (oversized || child_sum > w.stab_ptr[c]) {
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

            for (std::int32_t i = 0; i < cond_count; i++) {
                if (w.cond_c_ptr[i] >= w.row_count)
                    w.cprt_ptr[w.cond_c_ptr[i]] = w.cond_p_ptr[i];
            }

            // scikit-learn's epsilon_search. `is_ptr` bits: 0 EOM selection, 1 final pick,
            // 2 epsilon target.
            if (w.cluster_selection_epsilon > Float(0) && !w.is_ptr[root_cid]) {
                const Float eps = w.cluster_selection_epsilon;
                auto birth_dist = [&](std::int32_t c) {
                    return (w.lb_ptr[c] > Float(0)) ? Float(1) / w.lb_ptr[c] : Float(0);
                };
                for (std::int32_t c = root_cid + 1; c < n_clusters; c++) {
                    if (!(w.is_ptr[c] & 1))
                        continue;
                    if (!(birth_dist(c) < eps)) {
                        w.is_ptr[c] |= 2;
                        continue;
                    }
                    bool processed = false;
                    for (std::int32_t a = w.cprt_ptr[c]; a >= root_cid && a < n_clusters;
                         a = w.cprt_ptr[a]) {
                        if (w.is_ptr[a] & 4) {
                            processed = true;
                            break;
                        }
                        if (a == root_cid)
                            break;
                    }
                    if (processed)
                        continue;
                    std::int32_t target = c;
                    while (true) {
                        const std::int32_t parent = w.cprt_ptr[target];
                        if (parent < root_cid || parent >= n_clusters)
                            break;
                        if (parent == root_cid) {
                            if (w.allow_single_cluster)
                                target = root_cid;
                            break;
                        }
                        target = parent;
                        if (birth_dist(parent) > eps)
                            break;
                    }
                    w.is_ptr[target] |= 6;
                }
                for (std::int32_t c = root_cid; c < n_clusters; c++) {
                    w.is_ptr[c] = (w.is_ptr[c] & 2) ? 1 : 0;
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

/// Cluster extraction kernels 4 + 5: build the dendrogram parent links, then label each point
/// with its deepest selected ancestor cluster.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`)
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion of phase 5
template <typename Float>
inline sycl::event assign_label_kernels(sycl::queue& queue,
                                        const cluster_work_ptrs<Float>& w,
                                        const bk::event_vector& deps) {
    // Kernel 4: parent link of every node.
    auto k4_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(w.edge_count), [=](sycl::id<1> idx) {
            const std::int64_t nid = w.row_count + idx[0];
            const std::int32_t lc = w.lc_ptr[nid];
            const std::int32_t rc = w.rc_ptr[nid];
            if (lc >= 0)
                w.dp_ptr[lc] = static_cast<std::int32_t>(nid);
            if (rc >= 0)
                w.dp_ptr[rc] = static_cast<std::int32_t>(nid);
        });
    });

    // Kernel 5: label of every point.
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

            // The point fell out of a cluster: walk up to its deepest selected ancestor.
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

            // The point never fell out: walk up the dendrogram.
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

/// When only the root cluster is selected, demote to noise the points whose drop-out lambda is
/// below scikit-learn's threshold: `1 / epsilon` if epsilon is set, else the root's death lambda.
///
/// Does nothing unless the selection is the root alone.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`);
///                      `plam_ptr` must point at a `row_count`-long buffer
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion of the pass
template <typename Float>
inline sycl::event single_cluster_threshold_kernel(sycl::queue& queue,
                                                   const cluster_work_ptrs<Float>& w,
                                                   const bk::event_vector& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            const std::int32_t n_clusters = w.nc_ptr[0];
            if (n_clusters == 0)
                return;

            const std::int32_t root_cid = static_cast<std::int32_t>(w.row_count);
            if (!w.is_ptr[root_cid])
                return;
            for (std::int32_t c = root_cid + 1; c < n_clusters; c++) {
                if (w.is_ptr[c])
                    return;
            }

            const std::int32_t cond_count = w.cond_cnt_ptr[0];

            const Float inf_lambda = dal::detail::limits<Float>::max();

            Float threshold = Float(0);
            if (w.cluster_selection_epsilon > Float(0)) {
                // Cap the threshold at the zero-distance lambda, which `1 / epsilon` can exceed.
                threshold = (w.cluster_selection_epsilon > Float(1) / inf_lambda)
                                ? Float(1) / w.cluster_selection_epsilon
                                : inf_lambda;
            }
            else {
                for (std::int32_t ei = 0; ei < cond_count; ei++) {
                    if (w.cond_p_ptr[ei] == root_cid && w.cond_l_ptr[ei] > threshold)
                        threshold = w.cond_l_ptr[ei];
                }
            }

            // Points that never dropped out keep -1 and are demoted.
            for (std::int32_t ei = 0; ei < cond_count; ei++) {
                const std::int32_t child = w.cond_c_ptr[ei];
                if (child < static_cast<std::int32_t>(w.row_count))
                    w.plam_ptr[child] = w.cond_l_ptr[ei];
            }

            const std::int32_t root_label = w.clab_ptr[root_cid];
            for (std::int64_t i = 0; i < w.row_count; i++) {
                if (w.resp_ptr[i] == root_label && !(w.plam_ptr[i] >= threshold))
                    w.resp_ptr[i] = -1;
            }
        });
    });
}

/// Cluster extraction kernels 6-8: membership strength of every point, as in scikit-learn.
///
/// A point's strength is the lambda at which it dropped out, divided by the death lambda of its
/// cluster and clamped to 1; noise gets 0.
///
/// @tparam Float Floating-point type
///
/// @param[in]     queue The SYCL queue
/// @param[in,out] w     Working pointers and constants (see `cluster_work_ptrs`)
/// @param[in]     deps  Events that must complete before submission
///
/// @return Event signaling completion of phase 8
template <typename Float>
inline sycl::event membership_probability_kernels(sycl::queue& queue,
                                                  const cluster_work_ptrs<Float>& w,
                                                  const bk::event_vector& deps) {
    // Over the `3 * row_count` condensed-edge capacity `extract_clusters` allocates.
    auto k6_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(3 * w.row_count), [=](sycl::id<1> idx) {
            const std::int32_t ei = static_cast<std::int32_t>(idx[0]);
            if (ei >= w.cond_cnt_ptr[0])
                return;

            const std::int32_t parent = w.cond_p_ptr[ei];
            const std::int32_t child = w.cond_c_ptr[ei];
            const Float lambda = w.cond_l_ptr[ei];

            if (parent >= 0 && parent < static_cast<std::int32_t>(w.total_nodes))
                bk::atomic_global_max(w.cdeath_ptr + parent, lambda);
            if (child >= 0 && child < static_cast<std::int32_t>(w.row_count))
                w.plam_ptr[child] = lambda;
        });
    });

    auto k7_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k6_event });
        h.parallel_for(sycl::range<1>(w.total_nodes), [=](sycl::id<1> idx) {
            const std::int32_t c = static_cast<std::int32_t>(idx[0]);
            if (c < static_cast<std::int32_t>(w.row_count) || c >= w.nc_ptr[0])
                return;
            if (!w.is_ptr[c])
                return;
            const std::int32_t label = w.clab_ptr[c];
            if (label >= 0 && label < static_cast<std::int32_t>(w.total_nodes))
                w.ldeath_ptr[label] = w.cdeath_ptr[c];
        });
    });

    auto k8_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ k7_event });
        h.parallel_for(sycl::range<1>(w.row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t label = w.resp_ptr[i];
            if (label < 0 || label >= static_cast<std::int32_t>(w.total_nodes)) {
                w.prob_ptr[i] = Float(0);
                return;
            }

            const Float max_lambda = w.ldeath_ptr[label];
            const Float lambda = w.plam_ptr[i];
            if (!(max_lambda > Float(0)) || lambda < Float(0)) {
                w.prob_ptr[i] = Float(1);
                return;
            }
            w.prob_ptr[i] = (lambda < max_lambda) ? lambda / max_lambda : Float(1);
        });
    });

    return k8_event;
}

/// Extract HDBSCAN cluster labels from a sorted MST on the GPU.
///
/// @tparam Float Floating-point type used for MST weights
///
/// @param[in]  queue                     The SYCL queue
/// @param[in]  mst_from                  MST source endpoints, length `row_count - 1` (sorted by weight)
/// @param[in]  mst_to                    MST target endpoints, length `row_count - 1`
/// @param[in]  mst_weights               MST weights, length `row_count - 1` (ascending)
/// @param[out] responses                 Per-point cluster label, length `row_count` (-1 = noise)
/// @param[in]  row_count                 Number of points
/// @param[in]  min_cluster_size          Minimum cluster size (mcs)
/// @param[in]  deps                      Events that must complete before submission
/// @param[in]  cluster_selection         0 = EOM (default), 1 = leaf
/// @param[in]  allow_single_cluster      If false, reject root-only outcomes
/// @param[in]  cluster_selection_epsilon Distance epsilon for cluster_selection_epsilon (0 disables)
/// @param[in]  max_cluster_size          Maximum cluster size cap (0 disables)
/// @param[out] probabilities             Membership strength per point, length `row_count`, or
///                                       `nullptr` to skip
/// @param[out] single_linkage_tree       Dendrogram rows `[left, right, distance, size]`, length
///                                       `4 * (row_count - 1)`, or `nullptr` to skip
///
/// @return Event signaling completion of the labeling phase
template <typename Float>
inline sycl::event extract_clusters(sycl::queue& queue,
                                    const pr::ndview<std::int32_t, 1>& mst_from,
                                    const pr::ndview<std::int32_t, 1>& mst_to,
                                    const pr::ndview<Float, 1>& mst_weights,
                                    pr::ndview<std::int32_t, 1>& responses,
                                    std::int64_t row_count,
                                    std::int64_t min_cluster_size,
                                    const bk::event_vector& deps = {},
                                    std::int32_t cluster_selection = 0,
                                    bool allow_single_cluster = false,
                                    Float cluster_selection_epsilon = Float(0),
                                    std::int64_t max_cluster_size = 0,
                                    Float* probabilities = nullptr,
                                    Float* single_linkage_tree = nullptr) {
    ONEDAL_PROFILER_TASK(hdbscan.extract_clusters, queue);

    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_cluster_size >= 2);

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t total_nodes = 2 * row_count - 1;
    // At most `2 * (cluster_count - 1)` cluster edges plus `row_count` point edges, i.e.
    // `3 * row_count - 2`.
    const std::int64_t max_condensed = 3 * row_count;
    const std::int64_t max_clusters = total_nodes;

    // Allocate working memory on device
    auto [uf_parent, uf_parent_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_size, comp_size_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    auto [comp_to_node, comp_to_node_ev] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    queue.wait_and_throw();

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
    queue.wait_and_throw();

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
    queue.wait_and_throw();

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
    queue.wait_and_throw();

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
    queue.wait_and_throw();

    // Probability scratch, allocated only when requested; the per-point drop-out lambda is also
    // needed by the single-cluster threshold.
    const bool need_point_lambda = (probabilities != nullptr) || allow_single_cluster;
    pr::ndarray<Float, 1> cluster_death, label_death, point_lambda;
    if (probabilities != nullptr) {
        cluster_death = std::get<0>(
            pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device));
        label_death = std::get<0>(
            pr::ndarray<Float, 1>::zeros(queue, max_clusters, sycl::usm::alloc::device));
    }
    if (need_point_lambda) {
        point_lambda = std::get<0>(
            pr::ndarray<Float, 1>::full(queue, row_count, Float(-1), sycl::usm::alloc::device));
        queue.wait_and_throw();
    }

    bk::event_vector all_events = deps;
    all_events.insert(
        all_events.end(),
        { uf_parent_ev, comp_size_ev, comp_to_node_ev, ns_ev,   lc_ev,      rc_ev,       nw_ev,
          dtc_ev,       cp_ev,        cc_ev,           cl_ev,   cs_ev,      cca_ev,      stab_ev,
          lb_ev,        ilc_ev,       is_ev,           clab_ev, csz_ev,     cprt_ev,     cc0_ev,
          cc1_ev,       pff_ev,       dp_ev,           stk_ev,  stk_cid_ev, leaf_stk_ev, nca_ev });

    cluster_work_ptrs<Float> w;
    w.mst_from_ptr = mst_from.get_data();
    w.mst_to_ptr = mst_to.get_data();
    w.mst_weights_ptr = mst_weights.get_data();
    w.resp_ptr = responses.get_mutable_data();
    w.uf_parent_ptr = uf_parent.get_mutable_data();
    w.comp_size_ptr = comp_size.get_mutable_data();
    w.comp_to_node_ptr = comp_to_node.get_mutable_data();
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
    w.prob_ptr = probabilities;
    w.cdeath_ptr = (probabilities != nullptr) ? cluster_death.get_mutable_data() : nullptr;
    w.ldeath_ptr = (probabilities != nullptr) ? label_death.get_mutable_data() : nullptr;
    w.plam_ptr = need_point_lambda ? point_lambda.get_mutable_data() : nullptr;
    w.slt_ptr = single_linkage_tree;
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

    auto k2_event = build_dendrogram_kernels<Float>(queue, w, all_events);
    k2_event.wait_and_throw();

    // The dendrogram is final from here on; the later kernels only read it.
    if (single_linkage_tree != nullptr && edge_count > 0) {
        auto slt_event = dump_single_linkage_tree_kernel<Float>(queue, w, { k2_event });
        slt_event.wait_and_throw();
    }

    auto k3a_event = build_condensed_tree_kernel<Float>(queue, w, { k2_event });
    k3a_event.wait_and_throw();

    auto k3b_event = eom_select_clusters_kernel<Float>(queue, w, { k3a_event });
    k3b_event.wait_and_throw();

    auto k4_event = assign_label_kernels<Float>(queue, w, { k3b_event });
    k4_event.wait_and_throw();

    sycl::event labeling_event = k4_event;
    if (allow_single_cluster) {
        // `point_lambda` is freed on return and `sycl::free` does not wait.
        labeling_event = single_cluster_threshold_kernel<Float>(queue, w, { k4_event });
        labeling_event.wait_and_throw();
    }

    if (probabilities == nullptr) {
        return labeling_event;
    }

    // The probability scratch is freed on return and `sycl::free` does not wait.
    auto k8_event = membership_probability_kernels<Float>(queue, w, { labeling_event });
    k8_event.wait_and_throw();
    return k8_event;
}

#endif

} // namespace oneapi::dal::hdbscan::backend
