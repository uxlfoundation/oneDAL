/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <daal/src/algorithms/pca/pca_dense_svd_online_kernel.h>

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/partial_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using task_t = task::dim_reduction;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_pca = daal::algorithms::pca;
namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu>
using daal_svd_kernel_t = daal_pca::internal::PCASVDOnlineKernel<Float, Cpu>;

auto update_tables(const partial_train_input<task::dim_reduction>& input) {
    auto result = partial_train_result();
    const auto prev_ = input.get_prev();
    for (std::int64_t i = 0; i < prev_.get_auxiliary_table_count(); i++) {
        result.set_auxiliary_table(prev_.get_auxiliary_table(i));
    }
    return result;
}

template <typename Float>
static partial_train_result<task_t> call_daal_kernel_partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    const std::int64_t column_count = input.get_data().get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const auto input_ = input.get_prev();

    dal::detail::check_mul_overflow(column_count, column_count);

    const auto data = input.get_data();
    ONEDAL_ASSERT(data.has_data());

    const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);

    const bool has_nobs_data = input_.get_partial_n_rows().has_data();

    daal_pca::internal::InputDataType dtype = daal_pca::internal::nonNormalizedDataset;

    if (desc.get_data_normalization() == normalization::zscore) {
        dtype = daal_pca::internal::normalizedDataset;
    }

    if (has_nobs_data) {
        auto result = update_tables(input);
        auto daal_crossproduct_svd =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_crossproduct());
        auto daal_sums = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum());
        auto daal_nobs_matrix =
            interop::copy_to_daal_homogen_table<Float>(input_.get_partial_n_rows());
        auto auxiliaryTable = array<Float>::zeros(column_count * column_count);
        auto daal_auxiliary_svd = interop::convert_to_daal_homogen_table<Float>(auxiliaryTable,
                                                                                column_count,
                                                                                column_count);
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_svd_kernel_t>(ctx,
                                                                dtype,
                                                                daal_data,
                                                                *daal_nobs_matrix,
                                                                *daal_auxiliary_svd,
                                                                *daal_sums,
                                                                *daal_crossproduct_svd));
        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));
        result.set_partial_n_rows(
            interop::convert_from_daal_homogen_table<Float>(daal_nobs_matrix));
        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct_svd));

        result.set_auxiliary_table(
            interop::convert_from_daal_homogen_table<Float>(daal_auxiliary_svd));
        return result;
    }
    else {
        auto result = partial_train_result();
        auto arr_crossproduct_svd = array<Float>::zeros(column_count);
        auto arr_sums = array<Float>::zeros(column_count);
        auto arr_nobs_matrix = array<int>::zeros(1 * 1);
        auto auxiliaryTable = array<Float>::zeros(column_count * column_count);
        auto daal_crossproduct_svd =
            interop::convert_to_daal_homogen_table<Float>(arr_crossproduct_svd, 1, column_count);
        auto daal_auxiliary_svd = interop::convert_to_daal_homogen_table<Float>(auxiliaryTable,
                                                                                column_count,
                                                                                column_count);
        auto daal_sums = interop::convert_to_daal_homogen_table<Float>(arr_sums, 1, column_count);
        auto daal_nobs_matrix = interop::convert_to_daal_homogen_table<int>(arr_nobs_matrix, 1, 1);

        {
            interop::status_to_exception(
                interop::call_daal_kernel<Float, daal_svd_kernel_t>(ctx,
                                                                    dtype,
                                                                    daal_data,
                                                                    *daal_nobs_matrix,
                                                                    *daal_auxiliary_svd,
                                                                    *daal_sums,
                                                                    *daal_crossproduct_svd));
        }
        result.set_auxiliary_table(
            interop::convert_from_daal_homogen_table<Float>(daal_auxiliary_svd));

        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(daal_sums));

        result.set_partial_n_rows(interop::convert_from_daal_homogen_table<int>(daal_nobs_matrix));

        result.set_partial_crossproduct(
            interop::convert_from_daal_homogen_table<Float>(daal_crossproduct_svd));
        return result;
    }
}

template <typename Float>
static partial_train_result<task_t> partial_train(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_train_input<task::dim_reduction>& input) {
    return call_daal_kernel_partial_train<Float>(ctx, desc, input);
}

template <typename Float>
struct partial_train_kernel_cpu<Float, method::svd, task::dim_reduction> {
    partial_train_result<task_t> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_train_input<task::dim_reduction>& input) const {
        return partial_train<Float>(ctx, desc, input);
    }
};

template struct partial_train_kernel_cpu<float, method::svd, task::dim_reduction>;
template struct partial_train_kernel_cpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
