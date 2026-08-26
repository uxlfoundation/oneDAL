/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include <tuple>

namespace oneapi::dal::kmeans::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

/// Stores the information about data points that are candidates to fill the empty clusters
/// centers
///
/// @tparam Float The type of elements in the array that stores squared distances to the candidate
///         centorids.
template <typename Float>
class centroid_candidates {
public:
    /// Constructs the centorids candidates from the input arrays
    ///
    /// @param[in] indices               An array of size [c], where $c$ is the number candidates
    ///                                  to fill empty cluster centroids.
    ///                                  Value at i-th position indicates the index of the input data row
    ///                                  that would be taken as the i-th empty centroid candidate
    /// @param[in] distances             An array of size [c].
    ///                                  Value at i-th position indicates the squared distance between the
    ///                                  data point pointed by the 'indices' array
    ///                                  and the cluster centroid it belonged
    /// @param[in] empty_cluster_indices An array of size [c] that stores the row indices of the empty
    ///                                  cluster centers in the array of centroids
    /// @param[in] source_clusters       An array of size [c]. Value at i-th position indicates the cluster
    ///                                  from which the i-th candidate row was stolen (i.e. the cluster
    ///                                  the candidate row was assigned to before it became a centroid).
    ///                                  Filled locally by `find_candidates`; in the distributed path
    ///                                  `reduce_candidates` shuffles it in place so that slot i holds the
    ///                                  source cluster of the globally winning candidate for slot i.
    explicit centroid_candidates(const pr::ndarray<std::int32_t, 1>& indices,
                                 const pr::ndarray<Float, 1>& distances,
                                 const pr::ndarray<std::int32_t, 1>& empty_cluster_indices,
                                 const pr::ndarray<std::int32_t, 1>& source_clusters = {})
            : candidate_count_(indices.get_dimension(0)),
              indices_(indices),
              distances_(distances),
              empty_cluster_indices_(empty_cluster_indices),
              source_clusters_(source_clusters) {
        ONEDAL_ASSERT(candidate_count_ > 0);
        ONEDAL_ASSERT(empty_cluster_indices.get_dimension(0) == candidate_count_);
        ONEDAL_ASSERT(distances.get_dimension(0) == candidate_count_);
    }

    std::int64_t get_candidate_count() const {
        return candidate_count_;
    }

    const pr::ndarray<std::int32_t, 1>& get_indices() const {
        return indices_;
    }

    const pr::ndarray<Float, 1>& get_distances() const {
        return distances_;
    }

    const pr::ndarray<std::int32_t, 1>& get_empty_cluster_indices() const {
        return empty_cluster_indices_;
    }

    const pr::ndarray<std::int32_t, 1>& get_source_clusters() const {
        return source_clusters_;
    }

    void set_source_clusters(const pr::ndarray<std::int32_t, 1>& source_clusters) {
        source_clusters_ = source_clusters;
    }

private:
    std::int64_t candidate_count_;
    pr::ndarray<std::int32_t, 1> indices_;
    pr::ndarray<Float, 1> distances_;
    pr::ndarray<std::int32_t, 1> empty_cluster_indices_;
    pr::ndarray<std::int32_t, 1> source_clusters_;
};

template <typename Float>
auto find_candidates(sycl::queue& queue,
                     std::int64_t candidate_count,
                     const pr::ndarray<Float, 2>& closest_distances,
                     const pr::ndarray<std::int32_t, 1>& counters,
                     const pr::ndview<std::int32_t, 2>& responses,
                     const bk::event_vector& deps = {})
    -> std::tuple<centroid_candidates<Float>, sycl::event>;

/// Copies the data rows located at indices provided in candidates structure into
/// array of centroids
///
/// @tparam Float   The type of elements in the input data and centroids arrays.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param[in] values           An array of size [n + 1] of data values in the CSR layout,
///                             where $n$ is the number of rows in the input dataset
/// @param[in] column_indices   An array of column indices in the CSR layout
/// @param[in] row_offsets      An array of row offsets in the CSR layout
/// @param[in] candidates       Data structure that describes which input data rows should
///                             be copied to which positions in the centroids array
/// @param[in,out] centroids    An array of size [k x p], where $k$ is the number of centroids,
///                             $p$ is the number of features.
/// @param[in] deps             Events indicating availability of the input and output arrays
///                             for reading or writing
template <typename Float>
auto copy_candidates_from_data(sycl::queue& queue,
                               const pr::ndview<Float, 1>& values,
                               const pr::ndview<std::int64_t, 1>& column_indices,
                               const pr::ndview<std::int64_t, 1>& row_offsets,
                               const centroid_candidates<Float>& candidates,
                               pr::ndview<Float, 2>& centroids,
                               const bk::event_vector& deps) -> sycl::event;

/// Writes the winning candidate row into the centroid of each empty cluster. In the distributed
/// path the winners are agreed across ranks first; `counters` is needed there because the global
/// selection prefers candidates whose source cluster can spare a row (see `reduce_candidates`).
template <typename Float>
auto fill_empty_clusters(sycl::queue& queue,
                         bk::communicator<spmd::device_memory_access::usm>& comm,
                         const pr::ndview<Float, 2>& data,
                         centroid_candidates<Float>& candidates,
                         const pr::ndarray<std::int32_t, 1>& counters,
                         pr::ndview<Float, 2>& centroids,
                         const bk::event_vector& deps = {}) -> sycl::event;

/// Subtracts each newly-placed candidate row from its source cluster's centroid and decrements
/// the source cluster's counter. Called *after* `fill_empty_clusters` has written the winning
/// candidate row into `centroids[dst_i]` for each empty slot i, so this helper reads the stolen
/// row from `centroids[dst_i]` directly. This works uniformly for the single-rank dense,
/// distributed dense, and CSR paths — the fill writes the row into the same location regardless
/// of the source layout, so downstream doesn't need to know how the row arrived.
///
/// Without this correction, the source cluster's centroid stays at `sum_all / count_all` even
/// though one of its assigned points has been reassigned to an empty slot; this helper rewrites
/// it to `(sum_all - stolen) / (count_all - 1)`, which is what the *next* Lloyd iteration would
/// otherwise take an extra iteration to converge to.
///
/// Distributed handling: no extra communication is needed. `cluster_updater` allreduces both
/// `counters` and `centroids` *before* calling into empty-cluster handling, and
/// `reduce_candidates` leaves every rank with the same globally-agreed
/// (distance, row, source_cluster) tuples. Every rank therefore applies the identical correction
/// to identical inputs and stays in sync; the result must not be allreduced again, since that
/// would multiply the correction by the rank count.
///
/// Degenerate `count == 1` is skipped entirely -- neither the centroid nor the counter is touched,
/// since `(sum - stolen) / (count - 1)` is undefined there. Candidate selection demotes rows whose
/// cluster holds a single point (see `fill_candidate_indices_and_distances` and
/// `reduce_candidates`), so this branch is only reached when
///  * there are not enough other rows to fill every empty cluster, or
///  * two candidates come from the same two-point cluster, in which case the second one finds
///    `count == 1` after the first correction.
/// The source cluster then keeps `count == 1` and a centroid equal to the stolen row, which is a
/// valid data point; the empty cluster ends up with the same centroid, so the pair converges
/// without moving. The CPU kernel resolves the first case slightly differently -- it leaves the
/// empty cluster at its previous centroid rather than duplicating the source centroid -- because
/// it still has the previous centroids on hand, while here `centroids` has already been
/// overwritten with the newly computed values.
///
/// @tparam Float   The type of centroid elements.
///
/// @param[in]     queue        The DPC++ queue.
/// @param[in]     candidates   Structure describing candidate rows and their target empty-cluster
///                             slots. Must have `get_source_clusters()` populated -- filled by
///                             `find_candidates` and, in the distributed path, shuffled to match
///                             the global winners by `reduce_candidates`. If it is empty this
///                             helper is a no-op.
/// @param[in,out] centroids    The `[k x p]` centroids array; the stolen row is read from
///                             `centroids[empty_cluster_indices[i]]` and the source-cluster row
///                             is rewritten in place.
/// @param[in,out] counters     The `[k]` cluster counters; source cluster counts are decremented.
/// @param[in]     deps         Events that must complete before the correction runs.
template <typename Float>
inline auto correct_source_clusters(sycl::queue& queue,
                                    const centroid_candidates<Float>& candidates,
                                    pr::ndview<Float, 2>& centroids,
                                    pr::ndview<std::int32_t, 1>& counters,
                                    const bk::event_vector& deps = {}) -> sycl::event {
    const std::int64_t candidate_count = candidates.get_candidate_count();
    if (candidate_count == 0) {
        return sycl::event{};
    }
    const auto& source_clusters = candidates.get_source_clusters();
    if (!source_clusters.has_data()) {
        return sycl::event{};
    }

    const std::int64_t column_count = centroids.get_dimension(1);

    const std::int32_t* empty_cluster_indices_ptr =
        candidates.get_empty_cluster_indices().get_data();
    const std::int32_t* source_clusters_ptr = source_clusters.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    std::int32_t* counters_ptr = counters.get_mutable_data();

    // Deliberately a serial `single_task`: several candidates can be stolen from the same source
    // cluster, and the running `(sum - stolen) / (count - 1)` rewrite has to compose in order.
    // `candidate_count` is the number of empty clusters, so this loop is short by construction.
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.single_task([=]() {
            for (std::int64_t i = 0; i < candidate_count; ++i) {
                const std::int32_t src = source_clusters_ptr[i];
                // Defensive: responses always hold a valid cluster id, so this should not trigger.
                if (src < 0) {
                    continue;
                }
                const std::int32_t old_count = counters_ptr[src];
                if (old_count <= 1) {
                    continue;
                }
                const Float old_count_f = static_cast<Float>(old_count);
                const Float new_count_f = static_cast<Float>(old_count - 1);
                const std::int64_t dst = empty_cluster_indices_ptr[i];
                for (std::int64_t j = 0; j < column_count; ++j) {
                    const Float stolen = centroids_ptr[dst * column_count + j];
                    const Float sum = centroids_ptr[src * column_count + j] * old_count_f;
                    centroids_ptr[src * column_count + j] = (sum - stolen) / new_count_f;
                }
                counters_ptr[src] = old_count - 1;
            }
        });
    });
}

template <typename Float>
inline Float correct_objective_function(sycl::queue& queue,
                                        const centroid_candidates<Float>& candidates,
                                        const bk::event_vector& deps = {}) {
    sycl::event::wait_and_throw(deps);

    const auto& candidate_distances = candidates.get_distances();
    const std::int64_t candidate_count = candidate_distances.get_dimension(0);
    const auto host_candidate_distances = candidate_distances.to_host(queue);
    const Float* host_candidate_distances_ptr = host_candidate_distances.get_data();

    Float objective_function_correction = 0;
    for (std::int64_t i = 0; i < candidate_count; i++) {
        objective_function_correction -= host_candidate_distances_ptr[i];
    }

    return objective_function_correction;
}

/// Fills centroids that correspond to the empty clusters using dense input data
///
/// @param[in] queue              The DPC++ queue
/// @param[in] candidate_count    The number of empty clusters need to bu filled
/// @param[in] data               The [n x p] array of all feature vectors
/// @param[in] closest_distances  The distance between each observation and closest centroid,
///                               value at i-th position is $\min_j d(x_i, c_j)$, where $x_i$ is
///                               observation and $c_j$ is centroid
/// @param[in] counters           The number of observations assigned to each cluster,
///                               value at i-th position indicates that i-th clusters
///                               consists of `counters[i]` observations
/// @param[out] centroids         The centroids of [k x p], where $k$ is the number of centroids,
///                               $p$ is the number of features.
/// @param[in] deps               The vectors of events need to be completed before start computations
///
/// @return The correction coefficient needs to be added to the value of the objective function
template <typename Float>
inline auto handle_empty_clusters(sycl::queue& queue,
                                  bk::communicator<spmd::device_memory_access::usm>& comm,
                                  std::int64_t candidate_count,
                                  const pr::ndview<Float, 2>& data,
                                  const pr::ndview<std::int32_t, 2>& responses,
                                  const pr::ndarray<Float, 2>& closest_distances,
                                  pr::ndarray<std::int32_t, 1>& counters,
                                  pr::ndview<Float, 2>& centroids,
                                  const bk::event_vector& deps = {})
    -> std::tuple<Float, sycl::event> {
    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(data.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(counters.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(centroids.get_dimension(0) == counters.get_dimension(0));
    ONEDAL_ASSERT(centroids.get_dimension(1) == data.get_dimension(1));

    auto [candidates, find_candidates_event] =
        find_candidates(queue, candidate_count, closest_distances, counters, responses, deps);

    // `fill_empty_clusters` writes the winning candidate row into `centroids[dst_i]`. In the
    // distributed path it also shuffles `candidates.source_clusters_` inside `reduce_candidates`
    // so slot i names the source cluster of the globally winning candidate for slot i.
    auto fill_event = fill_empty_clusters(queue,
                                          comm,
                                          data,
                                          candidates,
                                          counters,
                                          centroids,
                                          { find_candidates_event });

    auto correct_event =
        correct_source_clusters(queue, candidates, centroids, counters, { fill_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, correct_event };
}

/// Fills centroids that correspond to the empty clusters using input data in CSR layout
///
/// @param[in] queue            The DPC++ queue
/// @param[in] values           An array of size [n + 1] of data values in the CSR layout,
///                             where $n$ is the number of rows in the input dataset
/// @param[in] column_indices   An array of column indices in the CSR layout
/// @param[in] row_offsets      An array of row offsets in the CSR layout
/// @param[in] row_count        A number of rows in the dataset
/// @param[in,out] centorids    The centroids of size [k x p], where $k$ is the number of centroids,
///                             $p$ is the number of features.
/// @param[in] candidate_count  The number of empty clusters need to bu filled
/// @param[in] cluster_counts   An array of size [k], where $k$ is the number of centroids, that stores
///                             number of observations assigned to each cluster.
///                             Value at i-th position indicates that i-th clusters
///                             consists of `cluster_counts[i]` observations.
/// @param[out] dists           An array of size [n], where $n$ is the number of rows in the input dataset,
///                             that stores the distances between each observation and closest centroid,
///                             value at i-th position is $\min_j d(x_i, c_j)$, where $x_i$ is i-th
///                             observation and $c_j$ is j-th centroid
/// @param[in] deps             Events indicating availability of the input and output arrays
///                             for reading or writing.
template <typename Float>
inline std::tuple<Float, sycl::event> handle_empty_clusters(
    sycl::queue& queue,
    const pr::ndview<Float, 1>& values,
    const pr::ndview<std::int64_t, 1>& column_indices,
    const pr::ndview<std::int64_t, 1>& row_offsets,
    const std::int64_t row_count,
    pr::ndarray<Float, 2>& centorids,
    const std::int64_t candidate_count,
    pr::ndarray<std::int32_t, 1>& cluster_counts,
    pr::ndarray<Float, 2>& dists,
    const pr::ndview<std::int32_t, 2>& responses,
    const bk::event_vector& deps = {}) {
    auto [candidates, find_candidates_event] =
        find_candidates(queue, candidate_count, dists, cluster_counts, responses, deps);

    // The CSR copy writes the sparse row expanded into a dense `centroids[dst_i]` row; the
    // correction below reads the row back from that location so it doesn't need a CSR-specific
    // path.
    auto copy_event = copy_candidates_from_data(queue,
                                                values,
                                                column_indices,
                                                row_offsets,
                                                candidates,
                                                centorids,
                                                { find_candidates_event });

    auto correct_event =
        correct_source_clusters(queue, candidates, centorids, cluster_counts, { copy_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, correct_event };
}

} // namespace oneapi::dal::kmeans::backend
