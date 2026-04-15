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

#include "oneapi/dal/algo/hdbscan/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/backend/cluster_utils.hpp"
#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::hdbscan::backend {

using dal::backend::context_cpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
static result_t compute_kernel_dense_impl(const context_cpu& ctx,
                                          const descriptor_t& desc,
                                          const table& data) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t col_count = data.get_column_count();
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int64_t min_samples = desc.get_min_samples();
    const std::int64_t edge_count = row_count - 1;

    row_accessor<const Float> accessor(data);
    const auto data_array = accessor.pull({ 0, -1 });
    const Float* data_ptr = data_array.get_data();

    // Step 1: Compute core distances
    std::vector<Float> core_dists(row_count);
    compute_core_distances_on_host(data_ptr, core_dists.data(), row_count, col_count, min_samples);

    // Step 2: Build MST using Prim's algorithm
    std::vector<std::int32_t> mst_from(edge_count);
    std::vector<std::int32_t> mst_to(edge_count);
    std::vector<Float> mst_weights(edge_count);
    build_mst_on_host(data_ptr,
                      core_dists.data(),
                      mst_from.data(),
                      mst_to.data(),
                      mst_weights.data(),
                      row_count,
                      col_count);

    // Step 3: Sort MST edges by weight
    sort_mst_by_weight_on_host(mst_from.data(), mst_to.data(), mst_weights.data(), edge_count);

    // Step 4: Extract flat clusters
    auto arr_responses = array<std::int32_t>::empty(row_count);
    std::int32_t* resp_ptr = arr_responses.get_mutable_data();

    const std::int64_t cluster_count = extract_clusters_from_mst(mst_from.data(),
                                                                 mst_to.data(),
                                                                 mst_weights.data(),
                                                                 resp_ptr,
                                                                 row_count,
                                                                 min_cluster_size);

    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(arr_responses, row_count, 1));
    }

    return results;
}

template <typename Float>
struct compute_kernel_cpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_dense_impl<Float>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_cpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_cpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
