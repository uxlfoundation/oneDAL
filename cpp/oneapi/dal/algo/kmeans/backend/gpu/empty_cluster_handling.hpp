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
    explicit centroid_candidates(const pr::ndarray<std::int32_t, 1>& indices,
                                 const pr::ndarray<Float, 1>& distances,
                                 const pr::ndarray<std::int32_t, 1>& empty_cluster_indices)
            : candidate_count_(indices.get_dimension(0)),
              indices_(indices),
              distances_(distances),
              empty_cluster_indices_(empty_cluster_indices) {
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

private:
    std::int64_t candidate_count_;
    pr::ndarray<std::int32_t, 1> indices_;
    pr::ndarray<Float, 1> distances_;
    pr::ndarray<std::int32_t, 1> empty_cluster_indices_;
};

template <typename Float>
auto find_candidates(sycl::queue& queue,
                     std::int64_t candidate_count,
                     const pr::ndarray<Float, 2>& closest_distances,
                     const pr::ndarray<std::int32_t, 1>& counters,
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

template <typename Float>
auto fill_empty_clusters(sycl::queue& queue,
                         bk::communicator<spmd::device_memory_access::usm>& comm,
                         const pr::ndview<Float, 2>& data,
                         const centroid_candidates<Float>& candidates,
                         pr::ndview<Float, 2>& centroids,
                         const bk::event_vector& deps = {}) -> sycl::event;

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
                                  const pr::ndarray<Float, 2>& closest_distances,
                                  const pr::ndarray<std::int32_t, 1>& counters,
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
        find_candidates(queue, candidate_count, closest_distances, counters, deps);

    auto fill_event =
        fill_empty_clusters(queue, comm, data, candidates, centroids, { find_candidates_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, fill_event };
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
    const bk::event_vector& deps = {}) {
    auto [candidates, find_candidates_event] =
        find_candidates(queue, candidate_count, dists, cluster_counts, deps);

    auto copy_event = copy_candidates_from_data(queue,
                                                values,
                                                column_indices,
                                                row_offsets,
                                                candidates,
                                                centorids,
                                                { find_candidates_event });

    const Float correction =
        correct_objective_function(queue, candidates, { find_candidates_event });

    return { correction, copy_event };
}

} // namespace oneapi::dal::kmeans::backend
