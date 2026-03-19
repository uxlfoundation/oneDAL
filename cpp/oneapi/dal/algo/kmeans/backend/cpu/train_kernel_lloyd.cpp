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

#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>
#include <daal/src/algorithms/kmeans/kmeans_lloyd_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/detail/train_init_centroids.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

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

template <typename Float, typename Method, typename Table>
static daal::data_management::NumericTablePtr get_initial_centroids(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const Table& data,
    const table& initial_centroids) {
    daal::data_management::NumericTablePtr daal_initial_centroids;
    if (!initial_centroids.has_data()) {
        daal_initial_centroids =
            oneapi::dal::kmeans::detail::daal_generate_centroids<Float, Method>(desc, data);
    }
    else {
        daal_initial_centroids = interop::convert_to_daal_table<Float>(initial_centroids);
    }
    return daal_initial_centroids;
}

inline auto get_daal_parameter_to_train(const descriptor_t& desc) {
    const std::int64_t cluster_count = desc.get_cluster_count();
    const std::int64_t max_iteration_count = desc.get_max_iteration_count();
    const double accuracy_threshold = desc.get_accuracy_threshold();

    daal_kmeans::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count),
                               dal::detail::integral_cast<std::size_t>(max_iteration_count));

    par.accuracyThreshold = accuracy_threshold;

    return par;
}

template <typename Float, typename Task, typename Method, typename Table>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const descriptor_t& desc,
                                           const Table& data,
                                           const table& initial_centroids) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = desc.get_cluster_count();

    auto par = get_daal_parameter_to_train(desc);

    auto daal_initial_centroids =
        get_initial_centroids<Float, Method>(ctx, desc, data, initial_centroids);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    auto result = train_result<Task>{};

    dal::detail::check_mul_overflow(cluster_count, column_count);

    array<Float> arr_centroids = array<Float>::empty(cluster_count * column_count);
    array<int> arr_iteration_count = array<int>::empty(1);

    const auto daal_centroids =
        interop::convert_to_daal_homogen_table(arr_centroids, cluster_count, column_count);
    const auto daal_iteration_count =
        interop::convert_to_daal_homogen_table(arr_iteration_count, 1, 1);

    daal::data_management::NumericTable* input[2] = { daal_data.get(),
                                                      daal_initial_centroids.get() };

    array<int> arr_responses = array<int>::empty(row_count);
    array<Float> arr_objective_function_value = array<Float>::empty(1);
    const auto daal_responses = interop::convert_to_daal_homogen_table(arr_responses, row_count, 1);
    const auto daal_objective_function_value =
        interop::convert_to_daal_homogen_table(arr_objective_function_value, 1, 1);

    daal::data_management::NumericTable* output[4] = { daal_centroids.get(),
                                                       daal_responses.get(),
                                                       daal_objective_function_value.get(),
                                                       daal_iteration_count.get() };

    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return batch_kernel_t<Float,
                              oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                              Method>()
            .compute(input, output, &par);
    }));

    result.set_objective_function_value(static_cast<double>(arr_objective_function_value[0]));

    result.set_responses(
        dal::detail::homogen_table_builder{}.reset(arr_responses, row_count, 1).build());

    result.set_iteration_count(static_cast<std::int64_t>(arr_iteration_count[0]));

    result.set_model(
        model<Task>().set_centroids(dal::detail::homogen_table_builder{}
                                        .reset(arr_centroids, cluster_count, column_count)
                                        .build()));
    return result;
}

template <typename Float, typename Task, typename Method>
static train_result<Task> train(const context_cpu& ctx,
                                const descriptor_t& desc,
                                const train_input<Task>& input) {
    using table_type =
        std::conditional_t<std::is_same_v<Method, method::lloyd_csr>, csr_table, table>;
    const auto data = static_cast<table_type>(input.get_data());
    return call_daal_kernel<Float, Task, Method>(ctx, desc, data, input.get_initial_centroids());
}

template <typename Float, typename Method>
struct train_kernel_cpu<Float, Method, task::clustering> {
    train_result<task::clustering> operator()(const context_cpu& ctx,
                                              const descriptor_t& desc,
                                              const train_input<task::clustering>& input) const {
        return train<Float, task::clustering, Method>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_cpu<double, method::lloyd_dense, task::clustering>;
template struct train_kernel_cpu<float, method::lloyd_csr, task::clustering>;
template struct train_kernel_cpu<double, method::lloyd_csr, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
