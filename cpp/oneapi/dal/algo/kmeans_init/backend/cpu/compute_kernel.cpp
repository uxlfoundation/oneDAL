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

#include <cmath>

#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/algo/kmeans_init/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/to_daal_method.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using std::int64_t;
using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::init>;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu, typename Method>
using daal_kmeans_init_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<to_daal_method<Method>::value, Float, Cpu>;

template <typename Float, typename Method, typename Task>
static compute_result<Task> call_daal_kernel(const context_cpu& ctx,
                                             const detail::descriptor_base<Task>& desc,
                                             const table& data) {
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = desc.get_cluster_count();

    //number of trials to pick each centroid from, 2 + int(ln(cluster_count)) works better than vanilla kmeans++
    //https://github.com/scikit-learn/scikit-learn/blob/a63b021310ba13ea39ad3555f550d8aeec3002c5/sklearn/cluster/_kmeans.py#L108
    std::int64_t trial_count = desc.get_local_trials_count();
    if (trial_count == -1) {
        const auto additional = std::log(cluster_count);
        trial_count = 2 + std::int64_t(additional);
    }

    daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count),
                                    0,
                                    dal::detail::integral_cast<std::size_t>(desc.get_seed()));
    par.nTrials = trial_count;

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const std::size_t len_input = 1;
    daal::data_management::NumericTable* input[len_input] = { daal_data.get() };

    dal::detail::check_mul_overflow(cluster_count, column_count);
    array<Float> arr_centroids = array<Float>::empty(cluster_count * column_count);
    const auto daal_centroids =
        interop::convert_to_daal_homogen_table(arr_centroids, cluster_count, column_count);
    const std::size_t len_output = 1;
    daal::data_management::NumericTable* output[len_output] = { daal_centroids.get() };

    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return daal_kmeans_init_kernel_t<
                   Float,
                   oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                   Method>()
            .compute(len_input,
                     input,
                     len_output,
                     output,
                     &par,
                     *daal::services::dynamicPointerCast<daal::algorithms::engines::BatchBase>(
                         par.engine));
    }));

    return compute_result<Task>().set_centroids(
        dal::detail::homogen_table_builder{}
            .reset(arr_centroids, cluster_count, column_count)
            .build());
}

template <typename Float, typename Method, typename Task>
static compute_result<Task> compute(const context_cpu& ctx,
                                    const descriptor_t& desc,
                                    const compute_input<Task>& input) {
    return call_daal_kernel<Float, Method, Task>(ctx, desc, input.get_data());
}

template <typename Float, typename Method, typename Task>
compute_result<Task> compute_kernel_cpu<Float, Method, Task>::operator()(
    const context_cpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const compute_input<Task>& input) const {
    return compute<Float, Method, Task>(ctx, desc, input);
}

template struct compute_kernel_cpu<float, method::dense, task::init>;
template struct compute_kernel_cpu<double, method::dense, task::init>;
template struct compute_kernel_cpu<float, method::random_dense, task::init>;
template struct compute_kernel_cpu<double, method::random_dense, task::init>;
template struct compute_kernel_cpu<float, method::random_csr, task::init>;
template struct compute_kernel_cpu<double, method::random_csr, task::init>;
template struct compute_kernel_cpu<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_cpu<double, method::plus_plus_dense, task::init>;
template struct compute_kernel_cpu<float, method::plus_plus_csr, task::init>;
template struct compute_kernel_cpu<double, method::plus_plus_csr, task::init>;
template struct compute_kernel_cpu<float, method::parallel_plus_dense, task::init>;
template struct compute_kernel_cpu<double, method::parallel_plus_dense, task::init>;
template struct compute_kernel_cpu<float, method::parallel_plus_csr, task::init>;
template struct compute_kernel_cpu<double, method::parallel_plus_csr, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
