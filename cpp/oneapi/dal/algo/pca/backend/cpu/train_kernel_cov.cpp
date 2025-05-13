/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/include/services/daal_defines.h>

#include <daal/src/algorithms/covariance/covariance_kernel.h>
#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>
#include <daal/src/algorithms/covariance/covariance_hyperparameter_impl.h>

#include "oneapi/dal/algo/pca/train_types.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/train_kernel_common.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;

using task_t = task::dim_reduction;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_pca_cor_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::batch, Float, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const detail::train_parameters<task_t>& params,
                                 const table& data) {
    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(row_count > 0);
    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);

    dal::detail::check_mul_overflow(column_count, component_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    auto arr_eigvec = array<Float>::empty(column_count * component_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);
    auto arr_means = array<Float>::empty(1 * column_count);
    auto arr_vars = array<Float>::empty(1 * column_count);
    auto arr_singular_values = array<Float>::empty(1 * component_count);
    auto arr_explained_variances_ratio = array<Float>::empty(1 * component_count);

    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, component_count, column_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);
    const auto daal_means = interop::convert_to_daal_homogen_table(arr_means, 1, column_count);
    const auto daal_variances = interop::convert_to_daal_homogen_table(arr_vars, 1, column_count);
    const auto daal_singular_values =
        interop::convert_to_daal_homogen_table(arr_singular_values, 1, component_count);
    const auto daal_explained_variances_ratio =
        interop::convert_to_daal_homogen_table(arr_explained_variances_ratio, 1, component_count);

    auto hp = convert_parameters<Float>(params);

    daal_cov::Batch<Float, daal_cov::defaultDense> covariance_alg;
    covariance_alg.input.set(daal_cov::data, daal_data);
    covariance_alg.setHyperparameter(&hp);

    daal::algorithms::pca::BaseBatchParameter daal_pca_parameter;

    daal_pca_parameter.isDeterministic = desc.get_deterministic();

    daal_pca_parameter.resultsToCompute = static_cast<DAAL_UINT64>(
        std::uint64_t(daal_pca::mean | daal_pca::variance | daal_pca::eigenvalue));

    daal_pca_parameter.isCorrelation = false;

    if (desc.get_normalization_mode() == normalization::mean_center) {
        daal_pca_parameter.doScale = false;
    }

    interop::status_to_exception(interop::call_daal_kernel<Float, daal_pca_cor_kernel_t>(
        ctx,
        *daal_data,
        &covariance_alg,
        *daal_eigenvectors,
        *daal_eigenvalues,
        *daal_means,
        *daal_variances,
        daal_singular_values.get(),
        daal_explained_variances_ratio.get(),
        &daal_pca_parameter));

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        result.set_eigenvectors(homogen_table::wrap(arr_eigvec, component_count, column_count));
    }

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(arr_eigval, 1, component_count));
    }

    if (desc.get_result_options().test(result_options::singular_values)) {
        result.set_singular_values(homogen_table::wrap(arr_singular_values, 1, component_count));
    }

    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        result.set_explained_variances_ratio(
            homogen_table::wrap(arr_explained_variances_ratio, 1, component_count));
    }

    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(homogen_table::wrap(arr_vars, 1, column_count));
    }

    if (desc.get_result_options().test(result_options::means)) {
        result.set_means(homogen_table::wrap(arr_means, 1, column_count));
    }

    return result;
}

template <typename Float>
static result_t train(const context_cpu& ctx,
                      const descriptor_t& desc,
                      const detail::train_parameters<task_t>& params,
                      const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, params, input.get_data());
}

template <typename Float>
struct train_kernel_cpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const detail::train_parameters<task_t>& params,
                        const input_t& input) const {
        return train<Float>(ctx, desc, params, input);
    }
};

template struct train_kernel_cpu<float, method::cov, task::dim_reduction>;
template struct train_kernel_cpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
