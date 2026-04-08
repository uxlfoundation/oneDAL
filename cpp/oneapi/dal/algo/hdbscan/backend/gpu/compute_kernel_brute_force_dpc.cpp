/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <limits>
#include <vector>

#include "oneapi/dal/algo/hdbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernels_fp.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data) {
    auto& queue = ctx.get_queue();

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t column_count = local_data.get_column_count();
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int64_t min_samples = desc.get_min_samples();
    const std::int64_t edge_count = row_count - 1;

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    // Step 1: Compute core distances
    auto [core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);

    auto core_event = kernels_fp<Float>::compute_core_distances(queue,
                                                                data_nd,
                                                                core_distances,
                                                                min_samples,
                                                                { core_dist_event });
    core_event.wait_and_throw();

    // Step 2: Build minimum spanning tree with mutual reachability distances
    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);

    auto mst_event =
        kernels_fp<Float>::build_mst(queue,
                                     data_nd,
                                     core_distances,
                                     mst_from,
                                     mst_to,
                                     mst_weights,
                                     { mst_from_event, mst_to_event, mst_weights_event });

    // Step 3: Sort MST edges by weight
    auto sort_event = kernels_fp<Float>::sort_mst_by_weight(queue,
                                                            mst_from,
                                                            mst_to,
                                                            mst_weights,
                                                            edge_count,
                                                            { mst_event });

    // Step 4: Extract flat clusters from the condensed tree
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);

    auto cluster_event = kernels_fp<Float>::extract_clusters(queue,
                                                             mst_from,
                                                             mst_to,
                                                             mst_weights,
                                                             arr_responses,
                                                             row_count,
                                                             min_cluster_size,
                                                             { sort_event, responses_event });
    cluster_event.wait_and_throw();

    // Count the number of clusters
    auto responses_host = arr_responses.to_host(queue);
    const auto* resp_ptr = responses_host.get_data();
    std::int32_t max_label = -1;
    for (std::int64_t i = 0; i < row_count; i++) {
        if (resp_ptr[i] > max_label) {
            max_label = resp_ptr[i];
        }
    }
    std::int64_t cluster_count = (max_label >= 0) ? (max_label + 1) : 0;

    // Build results
    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1));
    }

    if (desc.get_result_options().test(result_options::core_flags)) {
        // In HDBSCAN, core points are those with core_distance < infinity
        // (i.e., they have at least min_samples neighbors)
        auto [arr_core_flags, flags_event] =
            pr::ndarray<std::int32_t, 1>::zeros(queue, row_count, sycl::usm::alloc::device);

        auto core_dist_host = core_distances.to_host(queue);
        auto core_flags_host = pr::ndarray<std::int32_t, 1>::empty(row_count);
        const auto* cd_ptr = core_dist_host.get_data();
        auto* cf_ptr = core_flags_host.get_mutable_data();
        for (std::int64_t i = 0; i < row_count; i++) {
            cf_ptr[i] = (cd_ptr[i] < std::numeric_limits<Float>::max()) ? 1 : 0;
        }
        queue.memcpy(arr_core_flags.get_mutable_data(), cf_ptr, row_count * sizeof(std::int32_t))
            .wait_and_throw();
        results.set_core_flags(
            dal::homogen_table::wrap(arr_core_flags.flatten(queue), row_count, 1));
    }

    if (desc.get_result_options().test(result_options::core_observation_indices)) {
        // Find indices of core points (non-noise points)
        std::vector<std::int32_t> core_indices;
        for (std::int64_t i = 0; i < row_count; i++) {
            if (resp_ptr[i] >= 0) {
                core_indices.push_back(static_cast<std::int32_t>(i));
            }
        }
        std::int64_t core_count = static_cast<std::int64_t>(core_indices.size());
        if (core_count > 0) {
            auto arr_indices =
                pr::ndarray<std::int32_t, 1>::empty(queue, core_count, sycl::usm::alloc::device);
            queue
                .memcpy(arr_indices.get_mutable_data(),
                        core_indices.data(),
                        core_count * sizeof(std::int32_t))
                .wait_and_throw();
            results.set_core_observation_indices(
                dal::homogen_table::wrap(arr_indices.flatten(queue), core_count, 1));
        }
        else {
            results.set_core_observation_indices(dal::homogen_table{});
        }
    }

    if (desc.get_result_options().test(result_options::core_observations)) {
        // Gather core observation rows
        std::vector<std::int32_t> core_indices;
        for (std::int64_t i = 0; i < row_count; i++) {
            if (resp_ptr[i] >= 0) {
                core_indices.push_back(static_cast<std::int32_t>(i));
            }
        }
        std::int64_t core_count = static_cast<std::int64_t>(core_indices.size());
        if (core_count > 0) {
            auto core_obs = pr::ndarray<Float, 2>::empty(queue,
                                                         { core_count, column_count },
                                                         sycl::usm::alloc::device);
            // Copy row by row
            for (std::int64_t ci = 0; ci < core_count; ci++) {
                queue
                    .memcpy(core_obs.get_mutable_data() + ci * column_count,
                            data_nd.get_data() + core_indices[ci] * column_count,
                            column_count * sizeof(Float))
                    .wait_and_throw();
            }
            results.set_core_observations(
                dal::homogen_table::wrap(core_obs.flatten(queue), core_count, column_count));
        }
        else {
            results.set_core_observations(dal::homogen_table{});
        }
    }

    return results;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return compute_kernel_dense_impl<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_gpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
