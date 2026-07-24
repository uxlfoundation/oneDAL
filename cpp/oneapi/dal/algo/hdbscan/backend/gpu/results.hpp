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
#include "oneapi/dal/algo/hdbscan/backend/cluster_utils.hpp"
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
///
/// @return oneAPI `compute_result`
template <typename Float>
inline result_t make_results(sycl::queue& queue,
                             const descriptor_t& desc,
                             const pr::ndarray<std::int32_t, 1>& responses,
                             std::int64_t cluster_count) {
    const std::int64_t row_count = responses.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);

    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(responses.flatten(queue), row_count, 1));
    }

    return results;
}

/// Build a oneAPI compute result from device-side responses, including centers.
///
/// Forwards to the no-data overload to set responses and cluster count, then
/// optionally computes centroid and/or medoid tables on the host (via
/// `compute_centroids` / `compute_medoids`) when `desc.get_store_centers()`
/// requests them.
///
/// @tparam Float Floating-point type used for centers
///
/// @param[in] queue         The SYCL queue
/// @param[in] desc          Algorithm descriptor
/// @param[in] responses     Per-point cluster labels on device, length `n`
/// @param[in] cluster_count Number of distinct clusters
/// @param[in] data          Original input table (read for centers)
///
/// @return oneAPI `compute_result` with optional cluster/medoid centers
template <typename Float>
inline result_t make_results(sycl::queue& queue,
                             const descriptor_t& desc,
                             const pr::ndarray<std::int32_t, 1>& responses,
                             std::int64_t cluster_count,
                             const table& data) {
    auto results = make_results<Float>(queue, desc, responses, cluster_count);

    const auto store_centers = desc.get_store_centers();
    if (cluster_count > 0 && store_centers != store_centers_method::none &&
        desc.get_result_options().test(result_options::responses)) {
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t col_count = data.get_column_count();
        const bool need_centroids = (store_centers == store_centers_method::centroid ||
                                     store_centers == store_centers_method::both);
        const bool need_medoids = (store_centers == store_centers_method::medoid ||
                                   store_centers == store_centers_method::both);

        auto resp_host = responses.to_host(queue);
        const std::int32_t* resp_ptr = resp_host.get_data();

        row_accessor<const Float> data_acc{ data };
        const auto data_arr = data_acc.pull({ 0, -1 });
        const Float* data_ptr = data_arr.get_data();

        if (need_centroids) {
            auto arr_centroids = array<Float>::empty(cluster_count * col_count);
            compute_centroids(data_ptr,
                              resp_ptr,
                              row_count,
                              col_count,
                              cluster_count,
                              arr_centroids.get_mutable_data());
            results.set_cluster_centers(
                dal::homogen_table::wrap(arr_centroids, cluster_count, col_count));

            if (need_medoids) {
                auto arr_medoids = array<Float>::empty(cluster_count * col_count);
                compute_medoids(data_ptr,
                                resp_ptr,
                                row_count,
                                col_count,
                                cluster_count,
                                arr_centroids.get_data(),
                                arr_medoids.get_mutable_data());
                results.set_medoid_centers(
                    dal::homogen_table::wrap(arr_medoids, cluster_count, col_count));
            }
        }
        else if (need_medoids) {
            auto arr_centroids = array<Float>::empty(cluster_count * col_count);
            compute_centroids(data_ptr,
                              resp_ptr,
                              row_count,
                              col_count,
                              cluster_count,
                              arr_centroids.get_mutable_data());
            auto arr_medoids = array<Float>::empty(cluster_count * col_count);
            compute_medoids(data_ptr,
                            resp_ptr,
                            row_count,
                            col_count,
                            cluster_count,
                            arr_centroids.get_data(),
                            arr_medoids.get_mutable_data());
            results.set_medoid_centers(
                dal::homogen_table::wrap(arr_medoids, cluster_count, col_count));
        }
    }

    return results;
}

} // namespace oneapi::dal::hdbscan::backend
