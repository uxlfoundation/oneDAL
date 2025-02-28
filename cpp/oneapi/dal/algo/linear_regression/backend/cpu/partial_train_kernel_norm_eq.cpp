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

#include <daal/src/algorithms/linear_regression/linear_regression_train_kernel.h>
#include <daal/src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/partial_train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/train_kernel_common.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace be = dal::backend;
namespace interop = dal::backend::interop;
namespace daal_lr = daal::algorithms::linear_regression;

using daal_hyperparameters_t = daal_lr::internal::Hyperparameter;

constexpr auto daal_method = daal_lr::training::normEqDense;

template <typename Float, daal::CpuType Cpu>
using online_kernel_t = daal_lr::training::internal::OnlineKernel<Float, daal_method, Cpu>;

template <typename Float, typename Task>
static partial_train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                                   const detail::descriptor_base<Task>& desc,
                                                   const detail::train_parameters<Task>& params,
                                                   const partial_train_input<Task>& input) {
    using dal::detail::check_mul_overflow;

    const bool compute_intercept = desc.get_compute_intercept();

    const auto feature_count = input.get_data().get_column_count();
    const auto response_count = input.get_responses().get_column_count();

    const daal_hyperparameters_t& hp = convert_parameters<Float>(params);

    const auto ext_feature_count = feature_count + compute_intercept;

    const bool has_xtx_data = input.get_prev().get_partial_xtx().has_data();
    if (has_xtx_data) {
        auto daal_xtx =
            interop::copy_to_daal_homogen_table<Float>(input.get_prev().get_partial_xtx());
        auto daal_xty =
            interop::copy_to_daal_homogen_table<Float>(input.get_prev().get_partial_xty());
        auto x_daal_table = interop::convert_to_daal_table<Float>(input.get_data());
        auto y_daal_table = interop::convert_to_daal_table<Float>(input.get_responses());
        {
            const auto status = interop::call_daal_kernel<Float, online_kernel_t>(ctx,
                                                                                  *x_daal_table,
                                                                                  *y_daal_table,
                                                                                  *daal_xtx,
                                                                                  *daal_xty,
                                                                                  compute_intercept,
                                                                                  &hp);

            interop::status_to_exception(status);
        }
        auto result = partial_train_result<Task>();
        result.set_partial_xtx(interop::convert_from_daal_homogen_table<Float>(daal_xtx));
        result.set_partial_xty(interop::convert_from_daal_homogen_table<Float>(daal_xty));

        return result;
    }
    else {
        const auto xtx_size = check_mul_overflow(ext_feature_count, ext_feature_count);
        auto xtx_arr = array<Float>::zeros(xtx_size);

        const auto xty_size = check_mul_overflow(response_count, ext_feature_count);
        auto xty_arr = array<Float>::zeros(xty_size);

        auto xtx_daal_table =
            interop::convert_to_daal_homogen_table(xtx_arr, ext_feature_count, ext_feature_count);
        auto xty_daal_table =
            interop::convert_to_daal_homogen_table(xty_arr, response_count, ext_feature_count);

        auto x_daal_table = interop::convert_to_daal_table<Float>(input.get_data());
        auto y_daal_table = interop::convert_to_daal_table<Float>(input.get_responses());

        {
            const auto status = interop::call_daal_kernel<Float, online_kernel_t>(ctx,
                                                                                  *x_daal_table,
                                                                                  *y_daal_table,
                                                                                  *xtx_daal_table,
                                                                                  *xty_daal_table,
                                                                                  compute_intercept,
                                                                                  &hp);

            interop::status_to_exception(status);
        }

        auto result = partial_train_result<Task>();
        result.set_partial_xtx(interop::convert_from_daal_homogen_table<Float>(xtx_daal_table));
        result.set_partial_xty(interop::convert_from_daal_homogen_table<Float>(xty_daal_table));

        return result;
    }
}

template <typename Float, typename Task>
static partial_train_result<Task> train(const context_cpu& ctx,
                                        const detail::descriptor_base<Task>& desc,
                                        const detail::train_parameters<Task>& params,
                                        const partial_train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, params, input);
}

template <typename Float, typename Task>
struct partial_train_kernel_cpu<Float, method::norm_eq, Task> {
    partial_train_result<Task> operator()(const context_cpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const detail::train_parameters<Task>& params,
                                          const partial_train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct partial_train_kernel_cpu<float, method::norm_eq, task::regression>;
template struct partial_train_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
