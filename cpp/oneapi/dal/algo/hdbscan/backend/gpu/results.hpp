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
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

/// Copy a host array to a new device array.
///
/// @tparam T Element type
///
/// @param[in] queue The SYCL queue
/// @param[in] host  Host array
///
/// @return Device array with the same contents
template <typename T>
inline array<T> to_device_array(sycl::queue& queue, const array<T>& host) {
    return pr::ndarray<T, 1>::wrap(host).to_device(queue).flatten(queue);
}

/// Extract the flat clusters from an MST and pack them, with the outputs `desc` asks for, into a
/// oneAPI compute result.
///
/// The extraction walks the dendrogram and the condensed tree in order, which one CPU core does
/// far faster than one work-item, so it runs on host with the code of the CPU kernels. The
/// responses, probabilities and single linkage tree are returned on device.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue       The SYCL queue
/// @param[in] desc        Algorithm descriptor
/// @param[in] data        Input table, `n x d`, used for the centers
/// @param[in] mst_from    Source endpoint of each MST edge on host, length `n - 1`, any order
/// @param[in] mst_to      Target endpoint of each MST edge on host, length `n - 1`
/// @param[in] mst_weights Edge weights on host, length `n - 1`
///
/// @return oneAPI `compute_result`
template <typename Float>
inline result_t make_results_from_mst(sycl::queue& queue,
                                      const descriptor_t& desc,
                                      const table& data,
                                      const array<std::int32_t>& mst_from,
                                      const array<std::int32_t>& mst_to,
                                      const array<Float>& mst_weights) {
    ONEDAL_PROFILER_TASK(hdbscan.extract_clusters, queue);

    const std::int64_t row_count = data.get_row_count();
    const auto& options = desc.get_result_options();
    const bool need_centers = centers_requested(desc);
    const bool need_probabilities = options.test(result_options::probabilities) || need_centers;
    const bool need_single_linkage_tree = options.test(result_options::single_linkage_tree);

    const auto clusters = extract_clusters_from_mst<Float>(dal::backend::context_cpu{},
                                                           desc,
                                                           mst_from,
                                                           mst_to,
                                                           mst_weights,
                                                           row_count,
                                                           need_probabilities,
                                                           need_single_linkage_tree);

    auto results = result_t().set_cluster_count(clusters.cluster_count).set_result_options(options);

    if (options.test(result_options::responses)) {
        results.set_responses(
            homogen_table::wrap(to_device_array(queue, clusters.responses), row_count, 1));
    }
    if (options.test(result_options::probabilities)) {
        results.set_probabilities(
            homogen_table::wrap(to_device_array(queue, clusters.probabilities), row_count, 1));
    }
    // Left empty for a single observation, which has no merge to report.
    if (clusters.single_linkage_tree.get_count() > 0) {
        results.set_single_linkage_tree(
            homogen_table::wrap(to_device_array(queue, clusters.single_linkage_tree),
                                row_count - 1,
                                4));
    }
    if (need_centers && clusters.cluster_count > 0) {
        const auto data_host = row_accessor<const Float>(data).pull({ 0, -1 });
        set_cluster_centers(dal::backend::context_cpu{},
                            desc,
                            homogen_table::wrap(data_host, row_count, data.get_column_count()),
                            clusters.responses,
                            clusters.probabilities,
                            clusters.cluster_count,
                            results);
    }
    return results;
}

} // namespace oneapi::dal::hdbscan::backend
