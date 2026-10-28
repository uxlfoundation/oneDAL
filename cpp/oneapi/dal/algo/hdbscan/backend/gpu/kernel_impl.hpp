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

#include <numeric>
#include <tuple>
#include <vector>

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/block_tree.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/search/copy_search_callback.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

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
    const std::int64_t resident_bytes = mrd_mul_bytes(row_bytes, mrd_add_bytes(column_count, 1));

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

/// Whether the `n x n` MRD matrix and the buffers around it fit on the device of `queue`.
///
/// @tparam Float Floating-point type of the distance matrix
///
/// @param[in] queue        The SYCL queue whose device has to hold the matrix
/// @param[in] row_count    Number of input rows `n`
/// @param[in] column_count Number of input columns `d`
/// @param[in] min_samples  `k` of the core-distance k-selection
template <typename Float>
inline bool mrd_matrix_fits_device(sycl::queue& queue,
                                   std::int64_t row_count,
                                   std::int64_t column_count,
                                   std::int64_t min_samples) {
    constexpr std::int64_t element_size = static_cast<std::int64_t>(sizeof(Float));
    const std::int64_t max_alloc_bytes = bk::device_max_mem_alloc_size(queue);
    const std::int64_t global_mem_bytes = bk::device_global_mem_size(queue);

    // Size the matrix alone first: the scratch query needs an `n x n` shape to be representable.
    if (!mrd_matrix_fits_on_device(row_count,
                                   column_count,
                                   element_size,
                                   max_alloc_bytes,
                                   global_mem_bytes,
                                   min_samples,
                                   0)) {
        return false;
    }
    const std::int64_t kselect_scratch_bytes =
        pr::kselect_by_rows_scratch_size<Float>(queue, { row_count, row_count }, min_samples);
    return mrd_matrix_fits_on_device(row_count,
                                     column_count,
                                     element_size,
                                     max_alloc_bytes,
                                     global_mem_bytes,
                                     min_samples,
                                     kselect_scratch_bytes);
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
/// The rows are computed in blocks small enough for an int32 kernel range.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue  The SYCL queue
/// @param[in]  data   Input matrix of size `n x d`
/// @param[out] dist   Output matrix of size `n x n` (row-major)
/// @param[in]  metric Distance metric tag
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

    // Cosine needs the norms, so it keeps the primitive; the Lp metrics subtract coordinates.
    const auto by_row_blocks = [&](auto&& dist_op) {
        const bk::uniform_blocking blocking(n, bk::max_range_2d_rows(n));
        bk::event_vector block_events;
        for (std::int64_t b = 0; b < blocking.get_block_count(); b++) {
            const std::int64_t first = blocking.get_block_start_index(b);
            const std::int64_t last = blocking.get_block_end_index(b);
            auto dist_block = dist.get_row_slice(first, last);
            block_events.push_back(
                dist_op(data.get_row_slice(first, last), data, dist_block, deps));
        }
        return block_events;
    };

    switch (metric) {
        case distance_metric::cosine:
            return fix_zero_norm_cosine<Float>(queue,
                                               data,
                                               dist,
                                               by_row_blocks(pr::cosine_distance<Float>(queue)));
        default:
            return dispatch_lp_ops<Float>(metric, degree, [&](const auto& ops) {
                return compute_lp_distance_matrix<Float>(queue, data, ops, dist, deps);
            });
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

    // Not zero-filled: `kselect_by_rows` writes all `k` entries of every row.
    auto ksel_vals = pr::ndarray<Float, 2>::empty(queue, { n, k }, sycl::usm::alloc::device);

    pr::kselect_by_rows<Float> ksel(queue, { n, n }, k);
    auto ksel_event = ksel(queue, dist, k, ksel_vals, deps);

    const Float* ksel_ptr = ksel_vals.get_data();
    Float* core_ptr = core_distances.get_mutable_data();
    const bool needs_sqrt = (metric == distance_metric::euclidean);

    auto extract_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ksel_event });
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const Float val = ksel_ptr[i * k + (k - 1)];
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
/// @param[in]     alpha          Robust single-linkage scaling factor of `dist(i, j)`
/// @param[in]     alpha_scales_core Divide the core distances by alpha too, as scikit-learn's
///                                  brute-force path does
/// @param[in]     deps           Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float>
inline sycl::event compute_mrd_matrix(sycl::queue& queue,
                                      const pr::ndview<Float, 1>& core_distances,
                                      pr::ndview<Float, 2>& mrd_matrix,
                                      distance_metric metric,
                                      double alpha,
                                      bool alpha_scales_core,
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
    const Float core_scale = alpha_scales_core ? inv_alpha : Float(1);

    return bk::parallel_for_2d_by_row_blocks(
        queue,
        n,
        n,
        [=](std::int64_t i, std::int64_t j) {
            // Euclidean entries hold squared L2; the other metrics hold final distances.
            const Float d = needs_sqrt ? sycl::sqrt(sycl::fmax(mrd_ptr[i * n + j], Float(0)))
                                       : mrd_ptr[i * n + j];
            const Float cd_i = core_ptr[i] * core_scale;
            const Float cd_j = core_ptr[j] * core_scale;
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
    // One work-group per row, so the row is read coalesced.
    constexpr std::int64_t wg = 256;
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(bk::make_multiple_nd_range_1d(n * wg, wg), [=](sycl::nd_item<1> it) {
            constexpr std::int32_t none = std::numeric_limits<std::int32_t>::max();
            const auto group = it.get_group();
            const std::int64_t i = it.get_group(0);
            const std::int32_t my_comp = comp_ptr[i];
            const Float* row = mrd_ptr + i * n;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = none;
            for (std::int64_t j = it.get_local_id(0); j < n; j += wg) {
                if (comp_ptr[j] != my_comp && row[j] < best_m) {
                    best_m = row[j];
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            // The smallest MRD, then the lowest column among the lanes that reach it.
            const Float row_best = sycl::reduce_over_group(group, best_m, sycl::minimum<Float>());
            const std::int32_t row_best_j =
                sycl::reduce_over_group(group,
                                        (best_m == row_best) ? best_j : none,
                                        sycl::minimum<std::int32_t>());
            if (it.get_local_id(0) == 0) {
                pt_best_mrd_ptr[i] = row_best;
                pt_best_idx_ptr[i] = (row_best_j == none) ? -1 : row_best_j;
            }
        });
    });
}

/// The sequential part of the Boruvka rounds, run on host: the per-component reduction, the
/// union-find merge and the relabeling.
///
/// Each component takes its smallest MRD, ties going to its lowest row, and the components are
/// merged in id order, so every search that yields the same per-point bests gives the same MST.
///
/// @tparam Float Floating-point type of the MRD
template <typename Float>
class boruvka_merger {
public:
    using mst_t = std::tuple<array<std::int32_t>, array<std::int32_t>, array<Float>>;

    /// @param[in] row_count Number of points, at least 2
    explicit boruvka_merger(std::int64_t row_count)
            : n_(row_count),
              comp_(row_count),
              parent_(row_count),
              rank_(row_count, 0),
              best_mrd_(row_count),
              best_from_(row_count),
              component_count_(row_count) {
        ONEDAL_ASSERT(row_count > 1);
        std::iota(comp_.begin(), comp_.end(), 0);
        std::iota(parent_.begin(), parent_.end(), 0);
        from_ = array<std::int32_t>::empty(row_count - 1);
        to_ = array<std::int32_t>::empty(row_count - 1);
        weights_ = array<Float>::empty(row_count - 1);
    }

    /// Component of each point, a point id; refreshed by `merge`.
    const std::vector<std::int32_t>& components() const {
        return comp_;
    }

    std::int64_t component_count() const {
        return component_count_;
    }

    /// Append each component's best edge and merge its endpoints.
    ///
    /// @param[in] pt_best_mrd Best MRD of each point
    /// @param[in] pt_best_idx Best other-component point of each point, -1 if none
    ///
    /// @return Number of edges added
    std::int64_t merge(const Float* pt_best_mrd, const std::int32_t* pt_best_idx) {
        std::fill(best_mrd_.begin(), best_mrd_.end(), std::numeric_limits<Float>::max());
        std::fill(best_from_.begin(), best_from_.end(), -1);
        for (std::int64_t i = 0; i < n_; i++) {
            const std::int32_t c = comp_[i];
            if (pt_best_idx[i] >= 0 && pt_best_mrd[i] < best_mrd_[c]) {
                best_mrd_[c] = pt_best_mrd[i];
                best_from_[c] = static_cast<std::int32_t>(i);
            }
        }

        std::int32_t* from = from_.get_mutable_data();
        std::int32_t* to = to_.get_mutable_data();
        Float* weights = weights_.get_mutable_data();
        std::int64_t added = 0;
        for (std::int64_t c = 0; c < n_; c++) {
            const std::int32_t u = best_from_[c];
            if (u < 0) {
                continue;
            }
            const std::int32_t v = pt_best_idx[u];
            const std::int32_t ru = find(u);
            const std::int32_t rv = find(v);
            if (ru == rv) {
                continue;
            }
            from[edge_count_] = u;
            to[edge_count_] = v;
            weights[edge_count_] = best_mrd_[c];
            edge_count_++;
            added++;
            if (rank_[ru] < rank_[rv]) {
                parent_[ru] = rv;
            }
            else if (rank_[ru] > rank_[rv]) {
                parent_[rv] = ru;
            }
            else {
                parent_[rv] = ru;
                rank_[ru]++;
            }
        }
        for (std::int64_t i = 0; i < n_; i++) {
            comp_[i] = find(static_cast<std::int32_t>(i));
        }
        component_count_ -= added;
        return added;
    }

    /// The MST edges in the order they were added.
    ///
    /// @throws domain_error if non-finite input left the graph disconnected
    mst_t release() const {
        if (component_count_ > 1) {
            throw domain_error(dal::detail::error_messages::hdbscan_input_data_is_not_finite());
        }
        return { from_, to_, weights_ };
    }

private:
    std::int32_t find(std::int32_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    std::int64_t n_;
    std::vector<std::int32_t> comp_;
    std::vector<std::int32_t> parent_;
    std::vector<std::int32_t> rank_;
    std::vector<Float> best_mrd_;
    std::vector<std::int32_t> best_from_;
    std::int64_t component_count_;
    std::int64_t edge_count_ = 0;
    array<std::int32_t> from_;
    array<std::int32_t> to_;
    array<Float> weights_;
};

/// Run Boruvka rounds until one component is left or a round adds no edge.
///
/// @tparam Float Floating-point type
/// @tparam Find  Callable `(const std::int32_t* comp, Float* pt_best_mrd,
///               std::int32_t* pt_best_idx, sycl::event dep) -> sycl::event` running one round's
///               device search; the pointers are device USM of length `n`
///
/// @param[in] queue     The SYCL queue
/// @param[in] row_count Number of points, at least 2
/// @param[in] find      The device search
///
/// @return The MST edges on host
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float, typename Find>
inline typename boruvka_merger<Float>::mst_t run_boruvka(sycl::queue& queue,
                                                         std::int64_t row_count,
                                                         Find&& find) {
    const std::int64_t n = row_count;
    const auto alloc = sycl::usm::alloc::device;
    auto comp = pr::ndarray<std::int32_t, 1>::empty(queue, n, alloc);
    auto pt_best_mrd = pr::ndarray<Float, 1>::empty(queue, n, alloc);
    auto pt_best_idx = pr::ndarray<std::int32_t, 1>::empty(queue, n, alloc);
    std::vector<Float> pt_best_mrd_host(n);
    std::vector<std::int32_t> pt_best_idx_host(n);

    boruvka_merger<Float> merger(n);
    while (merger.component_count() > 1) {
        auto upload_event = queue.memcpy(comp.get_mutable_data(),
                                         merger.components().data(),
                                         n * sizeof(std::int32_t));
        auto find_event = find(comp.get_data(),
                               pt_best_mrd.get_mutable_data(),
                               pt_best_idx.get_mutable_data(),
                               upload_event);
        auto mrd_event = queue.memcpy(pt_best_mrd_host.data(),
                                      pt_best_mrd.get_data(),
                                      n * sizeof(Float),
                                      find_event);
        auto idx_event = queue.memcpy(pt_best_idx_host.data(),
                                      pt_best_idx.get_data(),
                                      n * sizeof(std::int32_t),
                                      find_event);
        sycl::event::wait_and_throw({ mrd_event, idx_event });
        if (merger.merge(pt_best_mrd_host.data(), pt_best_idx_host.data()) == 0) {
            break;
        }
    }
    return merger.release();
}

/// Build the MST under MRD with Boruvka's algorithm from a precomputed MRD matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue      The SYCL queue
/// @param[in] mrd_matrix Precomputed MRD matrix of size `n x n`, `n >= 2`
/// @param[in] deps       Events that must complete before the first round
///
/// @return The MST edges on host
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline typename boruvka_merger<Float>::mst_t build_mst(sycl::queue& queue,
                                                       const pr::ndview<Float, 2>& mrd_matrix,
                                                       const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst, queue);

    const std::int64_t n = mrd_matrix.get_dimension(0);
    ONEDAL_ASSERT(n > 1);
    ONEDAL_ASSERT(mrd_matrix.get_dimension(1) == n);

    sycl::event::wait_and_throw(deps);
    return run_boruvka<Float>(queue,
                              n,
                              [&](const std::int32_t* comp,
                                  Float* pt_best_mrd,
                                  std::int32_t* pt_best_idx,
                                  sycl::event dep) {
                                  return boruvka_find_nearest_mrd<Float>(queue,
                                                                         mrd_matrix.get_data(),
                                                                         comp,
                                                                         pt_best_mrd,
                                                                         pt_best_idx,
                                                                         n,
                                                                         { dep });
                              });
}

/// Core distance of every tree position, by the tree k-NN up to `block_tree_max_k` neighbors and
/// by `compute_core_distances_blocked` above it.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue       The SYCL queue
/// @param[in] data        Input `n x d`, in input order
/// @param[in] tree        Tree over `data`
/// @param[in] min_samples `k` of the core distance, at most `n`
/// @param[in] metric      Any metric but cosine
/// @param[in] degree      Minkowski degree
/// @param[in] block_size_hint `distance_block_size` of the fallback search, 0 for auto
///
/// @return Core distances in tree order; complete on return
template <typename Float>
inline pr::ndarray<Float, 1> compute_block_tree_core_distances(sycl::queue& queue,
                                                               const pr::ndview<Float, 2>& data,
                                                               const block_tree<Float>& tree,
                                                               std::int64_t min_samples,
                                                               distance_metric metric,
                                                               double degree,
                                                               std::int64_t block_size_hint) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_core_distances, queue);

    const std::int64_t n = tree.row_count;
    auto core = pr::ndarray<Float, 1>::empty(queue, n, sycl::usm::alloc::device);
    Float* core_ptr = core.get_mutable_data();

    if (min_samples <= block_tree_max_k) {
        auto event = dispatch_lp_ops<Float>(metric, degree, [&](const auto& ops) {
            if (min_samples <= 16) {
                return block_tree_core_distances<Float, 16>(queue,
                                                            tree,
                                                            ops,
                                                            min_samples,
                                                            core_ptr);
            }
            if (min_samples <= 32) {
                return block_tree_core_distances<Float, 32>(queue,
                                                            tree,
                                                            ops,
                                                            min_samples,
                                                            core_ptr);
            }
            if (min_samples <= 64) {
                return block_tree_core_distances<Float, 64>(queue,
                                                            tree,
                                                            ops,
                                                            min_samples,
                                                            core_ptr);
            }
            if (min_samples <= 128) {
                return block_tree_core_distances<Float, 128>(queue,
                                                             tree,
                                                             ops,
                                                             min_samples,
                                                             core_ptr);
            }
            return block_tree_core_distances<Float, block_tree_max_k>(queue,
                                                                      tree,
                                                                      ops,
                                                                      min_samples,
                                                                      core_ptr);
        });
        event.wait_and_throw();
        return core;
    }

    // `distance_block_size` if set, else about 256 MB of `block x n` distances.
    std::int64_t block_size = (block_size_hint > 0)
                                  ? block_size_hint
                                  : (256 * 1024 * 1024) / (n * std::int64_t(sizeof(Float)));
    block_size = std::max<std::int64_t>(1, std::min(block_size, n));
    auto core_input = pr::ndarray<Float, 1>::empty(queue, n, sycl::usm::alloc::device);
    auto core_event = compute_core_distances_blocked<Float>(queue,
                                                            data,
                                                            core_input,
                                                            min_samples,
                                                            block_size,
                                                            metric,
                                                            degree);
    const Float* core_input_ptr = core_input.get_data();
    const std::int32_t* original = tree.original.get_data();
    queue
        .submit([&](sycl::handler& h) {
            h.depends_on(core_event);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                core_ptr[idx[0]] = core_input_ptr[original[idx[0]]];
            });
        })
        .wait_and_throw();
    return core;
}

/// Build the MST under MRD with Boruvka's algorithm over a `block_tree`.
///
/// The device search gives the per-point bests of a full scan, ties included, so the MST is the
/// one `build_mst` would give on the matrix.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue     The SYCL queue
/// @param[in] tree      Tree over the input, at least 2 rows
/// @param[in] core      Core distance of each tree position, as it enters MRD
/// @param[in] metric    Any metric but cosine
/// @param[in] degree    Minkowski degree
/// @param[in] inv_alpha Scale of `dist(i, j)` inside MRD
///
/// @return The MST edges on host, endpoints as input rows
///
/// @throws domain_error if non-finite input leaves the graph disconnected
template <typename Float>
inline typename boruvka_merger<Float>::mst_t build_mst_block_tree(sycl::queue& queue,
                                                                  const block_tree<Float>& tree,
                                                                  const pr::ndview<Float, 1>& core,
                                                                  distance_metric metric,
                                                                  double degree,
                                                                  Float inv_alpha) {
    ONEDAL_PROFILER_TASK(hdbscan.build_mst, queue);

    const std::int64_t n = tree.row_count;
    const auto alloc = sycl::usm::alloc::device;
    auto tree_comp = pr::ndarray<std::int32_t, 1>::empty(queue, n, alloc);
    auto comp_best_mrd = pr::ndarray<Float, 1>::empty(queue, n, alloc);
    auto node_min_core = pr::ndarray<Float, 1>::empty(queue, tree.node_count(), alloc);
    auto node_comp = pr::ndarray<std::int32_t, 1>::empty(queue, tree.node_count(), alloc);
    std::int32_t* tree_comp_ptr = tree_comp.get_mutable_data();
    Float* comp_best_mrd_ptr = comp_best_mrd.get_mutable_data();
    const std::int32_t* original = tree.original.get_data();

    auto min_core_event = reduce_to_nodes(queue,
                                          tree,
                                          core.get_data(),
                                          node_min_core.get_mutable_data(),
                                          [](Float a, Float b) {
                                              return sycl::fmin(a, b);
                                          },
                                          {});

    auto mst = run_boruvka<Float>(
        queue,
        n,
        [&](const std::int32_t* comp,
            Float* pt_best_mrd,
            std::int32_t* pt_best_idx,
            sycl::event dep) {
            auto tree_comp_event = queue.submit([&](sycl::handler& h) {
                h.depends_on(dep);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    tree_comp_ptr[idx[0]] = comp[original[idx[0]]];
                    comp_best_mrd_ptr[idx[0]] = std::numeric_limits<Float>::max();
                });
            });
            auto node_comp_event = reduce_to_nodes(queue,
                                                   tree,
                                                   tree_comp.get_data(),
                                                   node_comp.get_mutable_data(),
                                                   [](std::int32_t a, std::int32_t b) {
                                                       return (a == b) ? a : -1;
                                                   },
                                                   { tree_comp_event });
            return dispatch_lp_ops<Float>(metric, degree, [&](const auto& ops) {
                auto search = [&](bool seed_only, const bk::event_vector& search_deps) {
                    return block_tree_boruvka_find<Float>(queue,
                                                          tree,
                                                          ops,
                                                          core.get_data(),
                                                          tree_comp.get_data(),
                                                          node_min_core.get_data(),
                                                          node_comp.get_data(),
                                                          inv_alpha,
                                                          comp_best_mrd_ptr,
                                                          pt_best_mrd,
                                                          pt_best_idx,
                                                          seed_only,
                                                          search_deps);
                };
                auto seed_event = search(true, { node_comp_event, min_core_event });
                return search(false, { seed_event });
            });
        });
    // The scratch above is freed on return; every round already waited for its search.
    return mst;
}

#endif

} // namespace oneapi::dal::hdbscan::backend
