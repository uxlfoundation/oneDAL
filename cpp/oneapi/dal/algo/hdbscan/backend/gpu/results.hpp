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
#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/algo/hdbscan/compute_types.hpp"
#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace pr = oneapi::dal::backend::primitives;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

/// Build a oneAPI compute result from device-side responses (no centers).
///
/// Wraps `responses` into a homogen_table when `result_options::responses` is
/// requested; always sets `cluster_count` and forwards `result_options`.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue         The SYCL queue
/// @param[in] desc          Algorithm descriptor (carries `result_options`)
/// @param[in] responses     Per-point cluster labels on device, length `n`
/// @param[in] cluster_count Number of distinct clusters
/// @param[in] probabilities Per-point membership strength on device, length `n`; empty
///                          unless `result_options::probabilities` was requested
/// @param[in] single_linkage_tree Dendrogram on device, length `4 * (n - 1)`; empty unless
///                          `result_options::single_linkage_tree` was requested
///
/// @return oneAPI `compute_result`
template <typename Float>
inline result_t make_results(sycl::queue& queue,
                             const descriptor_t& desc,
                             const pr::ndarray<std::int32_t, 1>& responses,
                             std::int64_t cluster_count,
                             const pr::ndarray<Float, 1>& probabilities = {},
                             const pr::ndarray<Float, 1>& single_linkage_tree = {}) {
    const std::int64_t row_count = responses.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);

    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(responses.flatten(queue), row_count, 1));
    }

    if (desc.get_result_options().test(result_options::probabilities)) {
        ONEDAL_ASSERT(probabilities.get_count() == row_count);
        results.set_probabilities(
            dal::homogen_table::wrap(probabilities.flatten(queue), row_count, 1));
    }

    // Left empty for a single observation, which has no merge to report.
    if (desc.get_result_options().test(result_options::single_linkage_tree) &&
        single_linkage_tree.get_count() > 0) {
        const std::int64_t edge_count = row_count - 1;
        ONEDAL_ASSERT(single_linkage_tree.get_count() == 4 * edge_count);
        results.set_single_linkage_tree(
            dal::homogen_table::wrap(single_linkage_tree.flatten(queue), edge_count, 4));
    }

    return results;
}

/// Build a oneAPI compute result from device-side responses, including centers.
///
/// Forwards to the no-data overload to set responses and cluster count, then fills the
/// centroid and/or medoid tables `desc.get_store_centers()` asks for, with scikit-learn's
/// probability-weighted formulas (see `set_cluster_centers`).
///
/// @tparam Float Floating-point type used for centers
///
/// @param[in] queue         The SYCL queue
/// @param[in] desc          Algorithm descriptor
/// @param[in] responses     Per-point cluster labels on device, length `n`
/// @param[in] cluster_count Number of distinct clusters
/// @param[in] data          Original input table (read for centers)
/// @param[in] probabilities Per-point membership strength on device, length `n`; empty
///                          unless `result_options::probabilities` was requested
/// @param[in] single_linkage_tree Dendrogram on device, length `4 * (n - 1)`; empty unless
///                          `result_options::single_linkage_tree` was requested
///
/// @return oneAPI `compute_result` with optional cluster/medoid centers
template <typename Float>
inline result_t make_results(sycl::queue& queue,
                             const descriptor_t& desc,
                             const pr::ndarray<std::int32_t, 1>& responses,
                             std::int64_t cluster_count,
                             const table& data,
                             const pr::ndarray<Float, 1>& probabilities = {},
                             const pr::ndarray<Float, 1>& single_linkage_tree = {}) {
    auto results = make_results<Float>(queue,
                                       desc,
                                       responses,
                                       cluster_count,
                                       probabilities,
                                       single_linkage_tree);

    const auto store_centers = desc.get_store_centers();
    if (cluster_count > 0 && store_centers != store_centers_method::none &&
        desc.get_result_options().test(result_options::responses)) {
        // The centers go through the host helper the CPU backends use, so both devices
        // produce identical centers; they only need the labels and the probabilities.
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t col_count = data.get_column_count();
        ONEDAL_ASSERT(probabilities.get_count() == row_count);
        const auto data_host = row_accessor<const Float>(data).pull({ 0, -1 });
        const auto responses_host = responses.to_host(queue);
        const auto probabilities_host = probabilities.to_host(queue);
        set_cluster_centers(dal::backend::context_cpu{},
                            desc,
                            data_host.get_data(),
                            responses_host.get_data(),
                            probabilities_host.get_data(),
                            row_count,
                            col_count,
                            cluster_count,
                            results);
    }

    return results;
}

} // namespace oneapi::dal::hdbscan::backend
