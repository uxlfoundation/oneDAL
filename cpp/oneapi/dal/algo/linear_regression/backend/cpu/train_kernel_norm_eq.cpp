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

#include <daal/src/algorithms/linear_regression/linear_regression_train_kernel.h>
#include <daal/src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h>
#include <daal/src/algorithms/ridge_regression/ridge_regression_train_kernel.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/train_kernel_common.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace be = dal::backend;
namespace pr = be::primitives;
namespace interop = dal::backend::interop;
namespace daal_lr = daal::algorithms::linear_regression;
namespace daal_rr = daal::algorithms::ridge_regression;

using daal_lr_hyperparameters_t = daal_lr::internal::Hyperparameter;

constexpr auto daal_lr_method = daal_lr::training::normEqDense;
constexpr auto daal_rr_method = daal_rr::training::normEqDense;

template <typename Float, daal::CpuType Cpu>
using batch_lr_kernel_t = daal_lr::training::internal::BatchKernel<Float, daal_lr_method, Cpu>;

template <typename Float, daal::CpuType Cpu>
using batch_rr_kernel_t = daal_rr::training::internal::BatchKernel<Float, daal_rr_method, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const detail::train_parameters<Task>& params,
                                           const table& data,
                                           const table& resp) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    const bool intp = desc.get_compute_intercept();

    const auto feature_count = data.get_column_count();
    const auto response_count = resp.get_column_count();

    const auto ext_feature_count = feature_count + intp;

    const auto xtx_size = check_mul_overflow(ext_feature_count, ext_feature_count);
    auto xtx_arr = array<Float>::zeros(xtx_size);

    const auto xty_size = check_mul_overflow(response_count, ext_feature_count);
    auto xty_arr = array<Float>::zeros(xty_size);

    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(betas_size);

    auto xtx_daal_table =
        interop::convert_to_daal_homogen_table(xtx_arr, ext_feature_count, ext_feature_count);
    auto xty_daal_table =
        interop::convert_to_daal_homogen_table(xty_arr, response_count, ext_feature_count);
    auto betas_daal_table =
        interop::convert_to_daal_homogen_table(betas_arr, response_count, feature_count + 1);

    auto x_daal_table = interop::convert_to_daal_table<Float>(data);
    auto y_daal_table = interop::convert_to_daal_table<Float>(resp);

    double alpha = desc.get_alpha();
    if (alpha != 0.0) {
        auto ridge_matrix_array = array<Float>::full(1, static_cast<Float>(alpha));
        auto ridge_matrix = interop::convert_to_daal_homogen_table<Float>(ridge_matrix_array, 1, 1);

        {
            const auto status =
                interop::call_daal_kernel<Float, batch_rr_kernel_t>(ctx,
                                                                    *x_daal_table,
                                                                    *y_daal_table,
                                                                    *xtx_daal_table,
                                                                    *xty_daal_table,
                                                                    *betas_daal_table,
                                                                    intp,
                                                                    *ridge_matrix);

            interop::status_to_exception(status);
        }
    }
    else {
        const daal_lr_hyperparameters_t& hp = convert_parameters<Float>(params);

        {
            const auto status =
                interop::call_daal_kernel<Float, batch_lr_kernel_t>(ctx,
                                                                    *x_daal_table,
                                                                    *y_daal_table,
                                                                    *xtx_daal_table,
                                                                    *xty_daal_table,
                                                                    *betas_daal_table,
                                                                    intp,
                                                                    &hp);

            interop::status_to_exception(status);
        }
    }

    auto betas_table = homogen_table::wrap(betas_arr, response_count, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(betas_table);
    const auto model = dal::detail::make_private<model_t>(model_impl);

    const auto options = desc.get_result_options();
    auto result = train_result<Task>().set_model(model).set_result_options(options);

    const pr::ndshape<2> betas_shape{ response_count, feature_count + 1 };
    auto betas = pr::ndarray<Float, 2>::wrap(betas_arr, betas_shape);

    if (options.test(result_options::intercept)) {
        auto arr = array<Float>::zeros(response_count);
        auto dst = pr::ndarray<Float, 2>::wrap_mutable(arr, { 1l, response_count });
        const auto src = betas.get_col_slice(0l, 1l).t();

        pr::copy(dst, src);

        auto intercept = homogen_table::wrap(arr, 1l, response_count);
        result.set_intercept(intercept);
    }

    if (options.test(result_options::coefficients)) {
        const auto size = check_mul_overflow(response_count, feature_count);

        auto arr = array<Float>::zeros(size);
        const auto src = betas.get_col_slice(1l, feature_count + 1);
        auto dst = pr::ndarray<Float, 2>::wrap_mutable(arr, { response_count, feature_count });

        pr::copy(dst, src);

        auto coefficients = homogen_table::wrap(arr, response_count, feature_count);
        result.set_coefficients(coefficients);
    }

    return result;
}

template <typename Float, typename Task>
static train_result<Task> train(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const detail::train_parameters<Task>& params,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx,
                                         desc,
                                         params,
                                         input.get_data(),
                                         input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct train_kernel_cpu<float, method::norm_eq, task::regression>;
template struct train_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
