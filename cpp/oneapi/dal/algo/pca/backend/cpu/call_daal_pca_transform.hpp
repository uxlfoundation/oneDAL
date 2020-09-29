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

#include <daal/src/algorithms/pca/transform/pca_transform_kernel.h>

#include "oneapi/dal/algo/pca/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/pca/infer_types.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = descriptor_base<task::dim_reduction>;

namespace daal_pca_tr = daal::algorithms::pca::transform;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_transform_kernel_t =
    daal_pca_tr::internal::TransformKernel<Float, daal_pca_tr::Method::defaultDense, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const model_t& model) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = desc.get_component_count();

    auto arr_data = row_accessor<const Float>{ data }.pull();
    auto arr_eigvec = row_accessor<const Float>{ model.get_eigenvectors() }.pull();

    auto arr_result = array<Float>::empty(row_count * component_count);

    // TODO: read-only access performed with deep copy of data since daal numeric tables are mutable.
    // Need to create special immutable homogen table on daal interop side

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, component_count, column_count);
    const auto daal_result =
        interop::convert_to_daal_homogen_table(arr_result, row_count, component_count);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_pca_transform_kernel_t>(ctx,
                                                                      *daal_data,
                                                                      *daal_eigenvectors,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      *daal_result));

    return result_t{}.set_transformed_data(
        dal::detail::homogen_table_builder{}.reset(arr_result, row_count, component_count).build());
}

template <typename Float>
static result_t infer(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

} // namespace oneapi::dal::pca::backend
