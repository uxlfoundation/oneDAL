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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
struct kernels_fp {
    /// Compute core distances using squared_l2_distance + kselect_by_rows primitives.
    /// Core distance = sqrt of k-th nearest squared L2 distance, where k = min_samples.
    static sycl::event compute_core_distances(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& data,
                                              pr::ndview<Float, 1>& core_distances,
                                              std::int64_t min_samples,
                                              const bk::event_vector& deps = {});

    /// Compute mutual reachability distance matrix on GPU.
    /// mrd(i,j) = max(core_dist[i], core_dist[j], euclidean_dist(i,j))
    static sycl::event compute_mrd_matrix(sycl::queue& queue,
                                          const pr::ndview<Float, 2>& data,
                                          const pr::ndview<Float, 1>& core_distances,
                                          pr::ndview<Float, 2>& mrd_matrix,
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
