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
    /// @param[in] source_clusters       An array of size [c]. Value at i-th position indicates the
    ///                                  cluster the i-th candidate row was assigned to before it
    ///                                  became a centroid. Filled by `find_candidates`; in the
    ///                                  distributed path `reduce_candidates` shuffles it to match the
    ///                                  globally winning candidates.
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
/// path the winners are agreed across ranks first (see `reduce_candidates`), purely by distance.
template <typename Float>
auto fill_empty_clusters(sycl::queue& queue,
                         bk::communicator<spmd::device_memory_access::usm>& comm,
                         const pr::ndview<Float, 2>& data,
                         centroid_candidates<Float>& candidates,
                         pr::ndview<Float, 2>& centroids,
                         const bk::event_vector& deps = {}) -> sycl::event;

/// Rewrites the centroid of every cluster a candidate row was taken from, from `sum / count` to
/// `(sum - stolen) / (count - 1)`, and decrements its counter. Runs after the fill has written the
/// candidate row into `centroids[dst_i]`, so the stolen row is read back from there and the dense,
/// distributed and CSR paths all share this helper.
///
/// A source cluster with `count <= 1` is left untouched, since `(sum - stolen) / (count - 1)` is
/// undefined: it keeps its point and its centroid, as the CPU kernels do. Candidates at distance
/// zero are skipped too - the fill did not move such a row, and `duplicate_largest_centroid`
/// handles those empty clusters.
///
/// Distributed: no extra communication. `counters` and `centroids` are allreduced before
/// empty-cluster handling starts and `reduce_candidates` leaves every rank with the same winning
/// tuples, so every rank applies the same correction and the result must not be allreduced again.
///
/// @tparam Float   The type of centroid elements.
///
/// @param[in]     queue        The DPC++ queue.
/// @param[in]     candidates   Candidate rows and their target empty-cluster slots. Needs
///                             `get_source_clusters()` populated; a no-op if it is empty.
/// @param[in,out] centroids    The `[k x p]` centroids; the source-cluster rows are rewritten
///                             in place.
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
    const Float* candidate_distances_ptr = candidates.get_distances().get_data();
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
                // The row was not moved, so there is nothing to take away from its cluster.
                if (!(candidate_distances_ptr[i] > Float(0))) {
                    continue;
                }
                const std::int32_t old_count = counters_ptr[src];
                if (old_count <= 1) {
                    continue;
                }
                // Evaluated as `centroid + (centroid - stolen) / (count - 1)`: the same value
                // without forming the `centroid * count` sum, so there is no cancellation for
                // large sums and the count only divides the correction term. The latter keeps
                // the result meaningful for counts past the mantissa width of `Float` (2^24 for
                // float32), where `count` and `count - 1` round to the same value.
                const Float inv_new_count = Float(1) / static_cast<Float>(old_count - 1);
                const std::int64_t dst = empty_cluster_indices_ptr[i];
                for (std::int64_t j = 0; j < column_count; ++j) {
                    const Float stolen = centroids_ptr[dst * column_count + j];
                    const Float centroid = centroids_ptr[src * column_count + j];
                    centroids_ptr[src * column_count + j] =
                        centroid + (centroid - stolen) * inv_new_count;
                }
                counters_ptr[src] = old_count - 1;
            }
        });
    });
}

/// Fills the empty clusters the fill declined to relocate - the ones whose candidate row already
/// sits on the centroid it is assigned to - with a duplicate of the centroid of the cluster holding
/// the most observations.
///
/// Such a cluster owns no points, so the choice does not move the objective function, but it has to
/// be a point of the data space rather than a leftover initial centroid to match what scikit-learn
/// returns for the same input. A duplicate is also stable: the assignment step gives tied points to
/// the lowest cluster index, so one of the pair stays empty and is refreshed to the same value on
/// the next iteration, leaving the centroid shift at zero.
///
/// Runs after `correct_source_clusters`, so `counters` and the donor centroid are this iteration's
/// final values, as in the CPU kernels. Needs no extra communication in the distributed path for
/// the same reason `correct_source_clusters` does not.
///
/// @tparam Float   The type of centroid elements.
///
/// @param[in]     queue        The DPC++ queue.
/// @param[in]     candidates   Candidate rows and their target empty-cluster slots.
/// @param[in]     counters     The `[k]` cluster counters, after the source-cluster correction.
/// @param[in,out] centroids    The `[k x p]` centroids array.
/// @param[in]     deps         Events that must complete before the fill runs.
template <typename Float>
inline auto duplicate_largest_centroid(sycl::queue& queue,
                                       const centroid_candidates<Float>& candidates,
                                       const pr::ndview<std::int32_t, 1>& counters,
                                       pr::ndview<Float, 2>& centroids,
                                       const bk::event_vector& deps = {}) -> sycl::event {
    const std::int64_t candidate_count = candidates.get_candidate_count();
    if (candidate_count == 0) {
        return sycl::event{};
    }

    const std::int64_t cluster_count = counters.get_dimension(0);
    const std::int64_t column_count = centroids.get_dimension(1);
    ONEDAL_ASSERT(centroids.get_dimension(0) == cluster_count);

    // The donor index is computed on the device to keep this off the host critical path; the
    // scan is over `cluster_count` counters, so a serial `single_task` is enough.
    auto donor = pr::ndarray<std::int32_t, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);

    const std::int32_t* counters_ptr = counters.get_data();
    std::int32_t* donor_ptr = donor.get_mutable_data();

    auto donor_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.single_task([=]() {
            std::int32_t best_cluster = -1;
            std::int32_t best_count = 0;
            for (std::int64_t i = 0; i < cluster_count; ++i) {
                // Strict `>`, so a tie goes to the lowest index - the same choice the CPU
                // kernels make.
                if (counters_ptr[i] > best_count) {
                    best_count = counters_ptr[i];
                    best_cluster = static_cast<std::int32_t>(i);
                }
            }
            donor_ptr[0] = best_cluster;
        });
    });

    const std::int32_t* empty_cluster_indices_ptr =
        candidates.get_empty_cluster_indices().get_data();
    const Float* candidate_distances_ptr = candidates.get_distances().get_data();
    Float* centroids_ptr = centroids.get_mutable_data();

    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(donor_event);
        cgh.parallel_for(bk::make_range_2d(candidate_count, column_count), [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            // A relocated cluster already holds its candidate row.
            if (candidate_distances_ptr[i] > Float(0)) {
                return;
            }
            const std::int32_t src = donor_ptr[0];
            // Not a single cluster holds an observation, so there is no centroid to duplicate.
            // Needs strictly fewer rows than clusters, which is rejected before training starts.
            if (src < 0) {
                return;
            }
            const std::int64_t dst = empty_cluster_indices_ptr[i];
            centroids_ptr[dst * column_count + j] = centroids_ptr[src * column_count + j];
        });
    });

    // `donor` is deallocated as this function returns, so the kernels reading it have to finish.
    fill_event.wait_and_throw();

    return fill_event;
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

    auto fill_event =
        fill_empty_clusters(queue, comm, data, candidates, centroids, { find_candidates_event });

    auto correct_event =
        correct_source_clusters(queue, candidates, centroids, counters, { fill_event });

    auto duplicate_event =
        duplicate_largest_centroid(queue, candidates, counters, centroids, { correct_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, duplicate_event };
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

    // The copy expands the sparse row into a dense `centroids[dst_i]` row, which is where
    // `correct_source_clusters` reads it back from.
    auto copy_event = copy_candidates_from_data(queue,
                                                values,
                                                column_indices,
                                                row_offsets,
                                                candidates,
                                                centorids,
                                                { find_candidates_event });

    auto correct_event =
        correct_source_clusters(queue, candidates, centorids, cluster_counts, { copy_event });

    auto duplicate_event =
        duplicate_largest_centroid(queue, candidates, cluster_counts, centorids, { correct_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, duplicate_event };
}

} // namespace oneapi::dal::kmeans::backend
