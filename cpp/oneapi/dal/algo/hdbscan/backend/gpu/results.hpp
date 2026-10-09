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
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernel_impl.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

/// Build a oneAPI compute result from device-side outputs, without centers.
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

/// Build a oneAPI compute result from device-side outputs, with the centers `store_centers` asks
/// for (see `set_cluster_centers`).
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

    if (cluster_count > 0 && centers_requested(desc)) {
        // The centers are computed on host, by the same kernel as on CPU.
        const std::int64_t row_count = data.get_row_count();
        ONEDAL_ASSERT(probabilities.get_count() == row_count);
        const auto data_host = row_accessor<const Float>(data).pull({ 0, -1 });
        auto responses_host = responses.to_host(queue);
        auto probabilities_host = probabilities.to_host(queue);
        set_cluster_centers(dal::backend::context_cpu{},
                            desc,
                            homogen_table::wrap(data_host, row_count, data.get_column_count()),
                            array<std::int32_t>::wrap(responses_host.get_mutable_data(), row_count),
                            array<Float>::wrap(probabilities_host.get_mutable_data(), row_count),
                            cluster_count,
                            results);
    }

    return results;
}

/// Extract the flat clusters from a weight-sorted MST and pack them, with the outputs `desc` asks
/// for, into a oneAPI compute result.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue       The SYCL queue
/// @param[in] desc        Algorithm descriptor
/// @param[in] data        Input table, `n x d`, used for the centers
/// @param[in] mst_from    Source endpoint of each MST edge, sorted by weight, length `n - 1`
/// @param[in] mst_to      Target endpoint of each MST edge, sorted by weight, length `n - 1`
/// @param[in] mst_weights Edge weights in ascending order, length `n - 1`
/// @param[in] deps        Events that must complete before the extraction
///
/// @return oneAPI `compute_result`
template <typename Float>
inline result_t make_results_from_sorted_mst(sycl::queue& queue,
                                             const descriptor_t& desc,
                                             const table& data,
                                             const pr::ndview<std::int32_t, 1>& mst_from,
                                             const pr::ndview<std::int32_t, 1>& mst_to,
                                             const pr::ndview<Float, 1>& mst_weights,
                                             const bk::event_vector& deps) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t edge_count = row_count - 1;
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int32_t cluster_selection =
        (desc.get_cluster_selection() == cluster_selection_method::leaf) ? 1 : 0;
    const bool allow_single_cluster = desc.get_allow_single_cluster();
    const Float cluster_selection_epsilon =
        static_cast<Float>(desc.get_cluster_selection_epsilon());
    const std::int64_t max_cluster_size = desc.get_max_cluster_size();

    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    responses_event.wait_and_throw();

    // Empty unless requested, which skips the probability kernels; the centers need them too.
    const bool need_probabilities =
        desc.get_result_options().test(result_options::probabilities) || centers_requested(desc);
    pr::ndarray<Float, 1> arr_probabilities;
    if (need_probabilities) {
        arr_probabilities =
            std::get<0>(pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device));
        queue.wait_and_throw();
    }

    // Empty unless requested; there is no merge below two rows.
    const bool need_single_linkage_tree =
        desc.get_result_options().test(result_options::single_linkage_tree) && edge_count > 0;
    pr::ndarray<Float, 1> arr_single_linkage_tree;
    if (need_single_linkage_tree) {
        arr_single_linkage_tree = std::get<0>(
            pr::ndarray<Float, 1>::zeros(queue, 4 * edge_count, sycl::usm::alloc::device));
        queue.wait_and_throw();
    }

    bk::event_vector cluster_deps = deps;
    cluster_deps.push_back(responses_event);
    auto cluster_event = extract_clusters<Float>(
        queue,
        mst_from,
        mst_to,
        mst_weights,
        arr_responses,
        row_count,
        min_cluster_size,
        cluster_deps,
        cluster_selection,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,
        need_probabilities ? arr_probabilities.get_mutable_data() : nullptr,
        need_single_linkage_tree ? arr_single_linkage_tree.get_mutable_data() : nullptr);
    cluster_event.wait_and_throw();

    // The cluster count is the largest label plus one.
    auto [max_label_arr, ml_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, -1, sycl::usm::alloc::device);
    sycl::event ml_alloc_ev = ml_ev;
    const std::int32_t* r_ptr = arr_responses.get_data();
    std::int32_t* ml_ptr = max_label_arr.get_mutable_data();
    auto reduce_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ml_alloc_ev });
        h.single_task([=]() {
            std::int32_t mx = -1;
            for (std::int64_t i = 0; i < row_count; i++) {
                if (r_ptr[i] > mx)
                    mx = r_ptr[i];
            }
            ml_ptr[0] = mx;
        });
    });
    reduce_event.wait_and_throw();
    auto max_label_host = max_label_arr.to_host(queue, { reduce_event });
    const std::int32_t max_label = max_label_host.get_data()[0];
    const std::int64_t cluster_count = (max_label >= 0) ? (max_label + 1) : 0;

    return make_results<Float>(queue,
                               desc,
                               arr_responses,
                               cluster_count,
                               data,
                               arr_probabilities,
                               arr_single_linkage_tree);
}

} // namespace oneapi::dal::hdbscan::backend
