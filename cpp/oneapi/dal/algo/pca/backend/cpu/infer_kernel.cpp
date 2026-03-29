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

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

namespace daal_pca_tr = daal::algorithms::pca::transform;
namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu>
using daal_pca_transform_kernel_t =
    daal_pca_tr::internal::TransformKernel<Float, daal_pca_tr::Method::defaultDense, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const model_t& model) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t component_count = get_component_count(desc, data);
    const auto sklearn_behavior = desc.get_normalization_mode() == normalization::mean_center;
    const auto sklearn_behavior_whiten = sklearn_behavior && desc.whiten();
    const auto daal_behavior_whiten =
        desc.get_normalization_mode() == normalization::zscore && desc.whiten();
    dal::detail::check_mul_overflow(row_count, component_count);
    auto arr_result = array<Float>::empty(row_count * component_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_eigenvectors = interop::convert_to_daal_table<Float>(model.get_eigenvectors());

    const auto daal_result =
        interop::convert_to_daal_homogen_table(arr_result, row_count, component_count);
    if (sklearn_behavior_whiten) {
        const auto daal_means = interop::convert_to_daal_table<Float>(model.get_means());
        const auto daal_eigenvalues =
            interop::convert_to_daal_table<Float>(model.get_eigenvalues());
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_transform_kernel_t>(ctx,
                                                                          *daal_data,
                                                                          *daal_eigenvectors,
                                                                          daal_means.get(),
                                                                          nullptr,
                                                                          daal_eigenvalues.get(),
                                                                          *daal_result));
    }
    else if (sklearn_behavior) {
        const auto daal_means = interop::convert_to_daal_table<Float>(model.get_means());
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_transform_kernel_t>(ctx,
                                                                          *daal_data,
                                                                          *daal_eigenvectors,
                                                                          daal_means.get(),
                                                                          nullptr,
                                                                          nullptr,
                                                                          *daal_result));
    }
    else if (daal_behavior_whiten) {
        const auto daal_means = interop::convert_to_daal_table<Float>(model.get_means());
        const auto daal_eigenvalues =
            interop::convert_to_daal_table<Float>(model.get_eigenvalues());
        const auto daal_variances = interop::convert_to_daal_table<Float>(model.get_variances());

        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_transform_kernel_t>(ctx,
                                                                          *daal_data,
                                                                          *daal_eigenvectors,
                                                                          daal_means.get(),
                                                                          daal_variances.get(),
                                                                          daal_eigenvalues.get(),
                                                                          *daal_result));
    }
    else {
        const auto daal_means = interop::convert_to_daal_table<Float>(model.get_means());
        const auto daal_variances = interop::convert_to_daal_table<Float>(model.get_variances());
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_pca_transform_kernel_t>(ctx,
                                                                          *daal_data,
                                                                          *daal_eigenvectors,
                                                                          daal_means.get(),
                                                                          daal_variances.get(),
                                                                          nullptr,
                                                                          *daal_result));
    }
    return result_t{}.set_transformed_data(
        dal::detail::homogen_table_builder{}.reset(arr_result, row_count, component_count).build());
}

template <typename Float>
static result_t infer(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float>
struct infer_kernel_cpu<Float, task::dim_reduction> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, task::dim_reduction>;
template struct infer_kernel_cpu<double, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
