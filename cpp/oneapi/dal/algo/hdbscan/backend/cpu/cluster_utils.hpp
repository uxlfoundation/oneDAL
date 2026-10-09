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
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::hdbscan::backend {

/// Fill the centroid and/or medoid tables `store_centers` asks for into `result`, computed by
/// the DAAL `HDBSCANCentersKernel` in scikit-learn's definitions.
///
/// Shared by the CPU backends and, on host, the GPU backend. Does nothing when there are no
/// clusters.
///
/// @tparam Float Floating-point type
///
/// @param[in]     ctx           CPU dispatch context
/// @param[in]     desc          Algorithm descriptor (store_centers, metric, degree)
/// @param[in]     data          Host input table, `row_count x col_count`
/// @param[in]     labels        Cluster id per point, length `row_count` (-1 = noise)
/// @param[in]     weights       Membership probability per point, length `row_count`
/// @param[in]     cluster_count Number of clusters
/// @param[in,out] result        Result receiving `cluster_centers` / `medoid_centers`
template <typename Float>
void set_cluster_centers(const dal::backend::context_cpu& ctx,
                         const detail::descriptor_base<task::clustering>& desc,
                         const table& data,
                         const array<std::int32_t>& labels,
                         const array<Float>& weights,
                         std::int64_t cluster_count,
                         compute_result<task::clustering>& result);

} // namespace oneapi::dal::hdbscan::backend
