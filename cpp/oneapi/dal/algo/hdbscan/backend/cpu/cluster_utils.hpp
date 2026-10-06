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

#include <cstdint>

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/algo/hdbscan/compute_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::hdbscan::backend {

/// Compute scikit-learn's centroid per cluster: the mean of its points weighted by their
/// membership probability. Points labeled noise or out of range are skipped.
///
/// @tparam Cpu   CPU dispatch tag (`backend::cpu_dispatch_*`)
/// @tparam Float Floating-point type
///
/// @param[in]  data          Row-major input buffer of size `row_count x col_count`
/// @param[in]  labels        Cluster id per point, length `row_count` (-1 = noise)
/// @param[in]  weights       Membership probability per point, length `row_count`
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[out] centroids     Row-major centroid buffer, size `cluster_count x col_count`
template <typename Cpu, typename Float>
void compute_centroids(const Float* data,
                       const std::int32_t* labels,
                       const Float* weights,
                       std::int64_t row_count,
                       std::int64_t col_count,
                       std::int64_t cluster_count,
                       Float* centroids);

/// Compute scikit-learn's medoid per cluster: the member `i` minimizing
/// `sum_j dist(i, j) * weight_j` over the members `j`, in the fitted metric. Ties go to the
/// lowest row index, as `np.argmin` does.
///
/// @tparam Cpu   CPU dispatch tag (`backend::cpu_dispatch_*`)
/// @tparam Float Floating-point type
///
/// @param[in]  data          Row-major input buffer of size `row_count x col_count`
/// @param[in]  labels        Cluster id per point, length `row_count` (-1 = noise)
/// @param[in]  weights       Membership probability per point, length `row_count`
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[in]  metric        Distance metric of the fit
/// @param[in]  degree        Minkowski degree, used only for `distance_metric::minkowski`
/// @param[out] medoids       Output medoid rows, size `cluster_count x col_count`
template <typename Cpu, typename Float>
void compute_medoids(const Float* data,
                     const std::int32_t* labels,
                     const Float* weights,
                     std::int64_t row_count,
                     std::int64_t col_count,
                     std::int64_t cluster_count,
                     distance_metric metric,
                     double degree,
                     Float* medoids);

/// Fill the centroid and/or medoid tables `store_centers` asks for into `result`.
///
/// Shared by the three CPU backends. Does nothing when there are no clusters.
///
/// @tparam Float Floating-point type
///
/// @param[in]     ctx           CPU dispatch context
/// @param[in]     desc          Algorithm descriptor (store_centers, metric, degree)
/// @param[in]     data          Row-major input buffer of size `row_count x col_count`
/// @param[in]     labels        Cluster id per point, length `row_count`
/// @param[in]     weights       Membership probability per point, length `row_count`
/// @param[in]     row_count     Number of input points
/// @param[in]     col_count     Number of features per point
/// @param[in]     cluster_count Number of clusters
/// @param[in,out] result        Result receiving `cluster_centers` / `medoid_centers`
template <typename Float>
void set_cluster_centers(const dal::backend::context_cpu& ctx,
                         const detail::descriptor_base<task::clustering>& desc,
                         const Float* data,
                         const std::int32_t* labels,
                         const Float* weights,
                         std::int64_t row_count,
                         std::int64_t col_count,
                         std::int64_t cluster_count,
                         compute_result<task::clustering>& result);

} // namespace oneapi::dal::hdbscan::backend
