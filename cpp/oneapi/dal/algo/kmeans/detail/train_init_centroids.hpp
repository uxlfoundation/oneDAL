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
#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/transfer.hpp"

namespace oneapi::dal::kmeans::detail {

namespace daal_kmeans_init = daal::algorithms::kmeans::init;

template <daal_kmeans_init::Method Value>
using daal_init_method_constant = std::integral_constant<daal_kmeans_init::Method, Value>;
using descriptor_t = detail::descriptor_base<task::clustering>;
namespace interop = dal::backend::interop;

template <typename Method>
struct to_daal_init_method;

template <>
struct to_daal_init_method<method::lloyd_dense>
        : daal_init_method_constant<daal_kmeans_init::plusPlusDense> {};

template <>
struct to_daal_init_method<method::lloyd_csr>
        : daal_init_method_constant<daal_kmeans_init::plusPlusCSR> {};

template <typename Float, daal::CpuType Cpu, typename Method>
using init_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<to_daal_init_method<Method>::value, Float, Cpu>;

template <typename Float, typename Method, typename Table>
inline daal::data_management::NumericTablePtr daal_generate_centroids(const descriptor_t& desc,
                                                                      const Table& data) {
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = desc.get_cluster_count();
    daal::data_management::NumericTablePtr daal_initial_centroids;
    const auto daal_data = interop::convert_to_daal_table<Float>(data, true);

    daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count));

    const std::size_t init_len_input = 1;
    const daal::data_management::NumericTable* init_input[init_len_input] = { daal_data.get() };

    daal_initial_centroids =
        interop::allocate_daal_homogen_table<Float>(cluster_count, column_count);
    const std::size_t init_len_output = 1;
    daal::data_management::NumericTable* init_output[init_len_output] = {
        daal_initial_centroids.get()
    };
    const dal::backend::context_cpu cpu_ctx;
    interop::status_to_exception(dal::backend::dispatch_by_cpu(cpu_ctx, [&](auto cpu) {
        return init_kernel_t<Float,
                             oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                             Method>()
            .compute(init_len_input, init_input, init_len_output, init_output, &par, *(par.engine));
    }));
    return daal_initial_centroids;
}

template daal::data_management::NumericTablePtr
daal_generate_centroids<float, method::lloyd_dense, table>(const descriptor_t& desc,
                                                           const table& data);
template daal::data_management::NumericTablePtr
daal_generate_centroids<double, method::lloyd_dense, table>(const descriptor_t& desc,
                                                            const table& data);
template daal::data_management::NumericTablePtr
daal_generate_centroids<float, method::lloyd_csr, csr_table>(const descriptor_t& desc,
                                                             const csr_table& data);
template daal::data_management::NumericTablePtr
daal_generate_centroids<double, method::lloyd_csr, csr_table>(const descriptor_t& desc,
                                                              const csr_table& data);

} // namespace oneapi::dal::kmeans::detail
