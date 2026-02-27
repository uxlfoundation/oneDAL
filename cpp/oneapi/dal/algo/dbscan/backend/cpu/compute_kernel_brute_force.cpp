/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/dbscan/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/dbscan/backend/fill_core_flags.hpp"

#include <daal/src/algorithms/dbscan/dbscan_kernel.h>

namespace oneapi::dal::dbscan::backend {

using dal::backend::context_cpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

namespace daal_dbscan = daal::algorithms::dbscan;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_dbscan_default_dense_t =
    daal_dbscan::internal::DBSCANBatchKernel<Float, daal_dbscan::defaultDense, Cpu>;

template <typename Float, daal::CpuType Cpu>
class dbscan_compute_wrapper {
public:
    template <typename... Args>
    auto compute(Args&&... args) {
        const daal_dbscan::Parameter* par =
            std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
        if (par->memorySavingMode == false) {
            return daal_dbscan::internal::DBSCANBatchKernel<Float, daal_dbscan::defaultDense, Cpu>{}
                .computeNoMemSave(std::forward<Args>(args)...);
        }
        return daal_dbscan::internal::DBSCANBatchKernel<Float, daal_dbscan::defaultDense, Cpu>{}
            .computeMemSave(std::forward<Args>(args)...);
    }
};

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& weights) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    const double epsilon = desc.get_epsilon();
    const std::int64_t min_observations = desc.get_min_observations();

    daal_dbscan::Parameter par(epsilon, dal::detail::integral_cast<std::size_t>(min_observations));
    par.memorySavingMode = desc.get_mem_save_mode();
    if (desc.get_result_options().test(result_options::core_observation_indices)) {
        par.resultsToCompute = daal_dbscan::computeCoreIndices;
    }
    else if (desc.get_result_options().test(result_options::core_observations)) {
        par.resultsToCompute = daal_dbscan::computeCoreObservations;
    }
    if (desc.get_result_options().test(result_options::core_observations)) {
        par.resultsToCompute |= daal_dbscan::computeCoreObservations;
    }

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_weights = interop::convert_to_daal_table<Float>(weights);

    array<int> arr_responses = array<int>::empty(row_count * 1);
    array<int> arr_cluster_count = array<int>::empty(1);

    const auto daal_responses = interop::convert_to_daal_homogen_table(arr_responses, row_count, 1);
    const auto daal_core_observation_indices = interop::empty_daal_homogen_table<int>(1);
    const auto daal_core_observations = interop::empty_daal_homogen_table<Float>(column_count);
    const auto daal_cluster_count = interop::convert_to_daal_homogen_table(arr_cluster_count, 1, 1);

    interop::status_to_exception(interop::call_daal_kernel<Float, dbscan_compute_wrapper>(
        ctx,
        daal_data.get(),
        daal_weights.get(),
        daal_responses.get(),
        daal_cluster_count.get(),
        daal_core_observation_indices.get(),
        daal_core_observations.get(),
        &par));

    auto core_observation_indices =
        interop::convert_from_daal_homogen_table<int>(daal_core_observation_indices);
    auto core_observations =
        interop::convert_from_daal_homogen_table<Float>(daal_core_observations);
    auto results = result_t()
                       .set_cluster_count(arr_cluster_count[0])
                       .set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(arr_responses, row_count, 1));
    }
    if (desc.get_result_options().test(result_options::core_flags)) {
        auto arr_core_flags = fill_core_flags(core_observation_indices, row_count);
        results.set_core_flags(dal::homogen_table::wrap(arr_core_flags, row_count, 1));
    }

    if (desc.get_result_options().test(result_options::core_observation_indices)) {
        results.set_core_observation_indices(core_observation_indices);
    }

    if (desc.get_result_options().test(result_options::core_observations)) {
        results.set_core_observations(core_observations);
    }

    return results;
}

template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto ret = call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_weights());
    return ret;
}

template <typename Float>
struct compute_kernel_cpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_cpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::dbscan::backend
