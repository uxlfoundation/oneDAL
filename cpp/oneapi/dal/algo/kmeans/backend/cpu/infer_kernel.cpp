/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/include/algorithms/kmeans/kmeans_types.h>
#include <daal/src/algorithms/kmeans/kmeans_lloyd_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans::backend {

using std::int64_t;
using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace daal_kmeans = daal::algorithms::kmeans;
namespace interop = dal::backend::interop;

template <daal_kmeans::Method Value>
using daal_method_constant = std::integral_constant<daal_kmeans::Method, Value>;

template <typename Method>
struct to_daal_method;

template <>
struct to_daal_method<method::lloyd_dense> : daal_method_constant<daal_kmeans::lloydDense> {};

template <>
struct to_daal_method<method::lloyd_csr> : daal_method_constant<daal_kmeans::lloydCSR> {};

template <typename Float, daal::internal::CpuType Cpu, typename Method>
using batch_kernel_t =
    daal_kmeans::internal::KMeansBatchKernel<to_daal_method<Method>::value, Float, Cpu>;

inline auto get_daal_parameter_to_infer(const descriptor_t& desc) {
    const std::int64_t max_iteration_count = 0;

    daal_kmeans::Parameter parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_cluster_count()),
        dal::detail::integral_cast<std::size_t>(max_iteration_count));

    if (desc.get_result_options().test(result_options::compute_exact_objective_function)) {
        parameter.resultsToEvaluate =
            static_cast<DAAL_UINT64>(daal_kmeans::computeAssignments) |
            static_cast<DAAL_UINT64>(daal_kmeans::computeExactObjectiveFunction);
    }
    else {
        parameter.resultsToEvaluate = static_cast<DAAL_UINT64>(daal_kmeans::computeAssignments);
    }
    return parameter;
}

template <typename Float, typename Task, typename Method, typename Table>
static infer_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const descriptor_t& desc,
                                           const model<Task>& trained_model,
                                           const Table& data) {
    const std::int64_t row_count = data.get_row_count();

    auto result = infer_result<Task>{}.set_result_options(desc.get_result_options());

    auto par = get_daal_parameter_to_infer(desc);

    array<int> arr_responses = array<int>::empty(row_count);
    array<Float> arr_objective_function_value = array<Float>::empty(1);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_initial_centroids =
        interop::convert_to_daal_table<Float>(trained_model.get_centroids());
    const auto daal_responses = interop::convert_to_daal_homogen_table(arr_responses, row_count, 1);
    const auto daal_objective_function_value =
        interop::convert_to_daal_homogen_table(arr_objective_function_value, 1, 1);

    daal::data_management::NumericTable* input[2] = { daal_data.get(),
                                                      daal_initial_centroids.get() };

    daal::data_management::NumericTable* output[4] = { nullptr,
                                                       daal_responses.get(),
                                                       daal_objective_function_value.get(),
                                                       nullptr };

    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return batch_kernel_t<Float,
                              oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                              Method>()
            .compute(input, output, &par);
    }));

    // Responses or assignments are always set
    result.set_responses(
        dal::detail::homogen_table_builder{}.reset(arr_responses, row_count, 1).build());

    // Objective function value is set only if it was requested
    if (desc.get_result_options().test(result_options::compute_exact_objective_function)) {
        result.set_objective_function_value(static_cast<double>(arr_objective_function_value[0]));
    }

    return result;
}

template <typename Float, typename Task, typename Method>
static infer_result<Task> infer(const context_cpu& ctx,
                                const descriptor_t& desc,
                                const infer_input<Task>& input) {
    using table_type =
        std::conditional_t<std::is_same_v<Method, method::lloyd_csr>, csr_table, table>;
    const auto data = static_cast<table_type>(input.get_data());
    return call_daal_kernel<Float, Task, Method>(ctx, desc, input.get_model(), data);
}

template <typename Float, typename Method>
struct infer_kernel_cpu<Float, Method, task::clustering> {
    infer_result<task::clustering> operator()(const context_cpu& ctx,
                                              const descriptor_t& desc,
                                              const infer_input<task::clustering>& input) const {
        return infer<Float, task::clustering, Method>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::lloyd_csr, task::clustering>;
template struct infer_kernel_cpu<double, method::lloyd_csr, task::clustering>;
template struct infer_kernel_cpu<float, method::lloyd_dense, task::clustering>;
template struct infer_kernel_cpu<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
