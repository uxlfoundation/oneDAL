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

/// True iff `desc` asks for cluster centers, which are computed from the responses.
///
/// @param[in] desc Algorithm descriptor
inline bool centers_requested(const detail::descriptor_base<task::clustering>& desc) {
    return desc.get_store_centers() != store_centers_method::none &&
           desc.get_result_options().test(result_options::responses);
}

/// Host outputs of `extract_clusters_from_mst`; the optional arrays are empty unless requested.
template <typename Float>
struct extracted_clusters {
    array<std::int32_t> responses;
    array<Float> probabilities;
    array<Float> single_linkage_tree;
    std::int64_t cluster_count = 0;
};

/// Extract the flat clusters from an MST on host, with the same code as the CPU kernels.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx                 CPU dispatch context
/// @param[in] desc                Algorithm descriptor
/// @param[in] mst_from            Source endpoint of each MST edge, length `row_count - 1`
/// @param[in] mst_to              Target endpoint of each MST edge, length `row_count - 1`
/// @param[in] mst_weights         Edge weights, length `row_count - 1`, in any order
/// @param[in] row_count           Number of points
/// @param[in] need_probabilities  Also compute the membership strengths
/// @param[in] need_single_linkage Also export the single linkage tree
///
/// @return Labels, cluster count and the requested optional outputs
template <typename Float>
extracted_clusters<Float> extract_clusters_from_mst(
    const dal::backend::context_cpu& ctx,
    const detail::descriptor_base<task::clustering>& desc,
    const array<std::int32_t>& mst_from,
    const array<std::int32_t>& mst_to,
    const array<Float>& mst_weights,
    std::int64_t row_count,
    bool need_probabilities,
    bool need_single_linkage);

/// Fill the centroid and/or medoid tables `store_centers` asks for into `result`, as
/// scikit-learn defines them. Does nothing when there are no clusters.
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
