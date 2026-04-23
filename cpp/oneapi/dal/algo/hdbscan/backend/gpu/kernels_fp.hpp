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
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
struct kernels_fp {
    /// Compute full pairwise distance matrix.
    /// For euclidean: dist²[i,j] = ||x_i||² + ||x_j||² - 2 * x_i · x_j (squared L2)
    /// For other metrics: actual distances computed directly.
    static sycl::event compute_distance_matrix(sycl::queue& queue,
                                               const pr::ndview<Float, 2>& data,
                                               pr::ndview<Float, 2>& dist,
                                               distance_metric metric,
                                               double degree,
                                               const bk::event_vector& deps = {});

    /// Compute core distances from pre-computed distance matrix via kselect.
    /// For euclidean: takes sqrt of k-th nearest squared distance.
    /// For other metrics: k-th nearest distance is used directly.
    static sycl::event compute_core_distances(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& dist,
                                              pr::ndview<Float, 1>& core_distances,
                                              std::int64_t min_samples,
                                              std::int64_t row_count,
                                              distance_metric metric,
                                              const bk::event_vector& deps = {});

    /// Transform pre-computed distance matrix into mutual reachability distances.
    /// mrd(i,j) = max(core_dist[i], core_dist[j], dist(i,j))
    /// For euclidean: applies sqrt to convert squared L2 to Euclidean first.
    static sycl::event compute_mrd_matrix(sycl::queue& queue,
                                          const pr::ndview<Float, 1>& core_distances,
                                          pr::ndview<Float, 2>& mrd_matrix,
                                          distance_metric metric,
                                          const bk::event_vector& deps = {});

    /// Build MST using Prim's algorithm on GPU with precomputed MRD matrix.
    static sycl::event build_mst(sycl::queue& queue,
                                 const pr::ndview<Float, 2>& mrd_matrix,
                                 pr::ndview<std::int32_t, 1>& mst_from,
                                 pr::ndview<std::int32_t, 1>& mst_to,
                                 pr::ndview<Float, 1>& mst_weights,
                                 std::int64_t row_count,
                                 const bk::event_vector& deps = {});

    /// Sort MST edges by weight using radix_sort_indices_inplace primitive.
    static sycl::event sort_mst_by_weight(sycl::queue& queue,
                                          pr::ndview<std::int32_t, 1>& mst_from,
                                          pr::ndview<std::int32_t, 1>& mst_to,
                                          pr::ndview<Float, 1>& mst_weights,
                                          std::int64_t edge_count,
                                          const bk::event_vector& deps = {});

    /// Extract flat clusters using EOM on device. Labels: -1 = noise.
    static sycl::event extract_clusters(sycl::queue& queue,
                                        const pr::ndview<std::int32_t, 1>& mst_from,
                                        const pr::ndview<std::int32_t, 1>& mst_to,
                                        const pr::ndview<Float, 1>& mst_weights,
                                        pr::ndview<std::int32_t, 1>& responses,
                                        std::int64_t row_count,
                                        std::int64_t min_cluster_size,
                                        const bk::event_vector& deps = {});
};

#endif

} // namespace oneapi::dal::hdbscan::backend
