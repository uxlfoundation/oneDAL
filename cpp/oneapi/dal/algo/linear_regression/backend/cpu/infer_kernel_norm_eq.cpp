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

#include <daal/src/algorithms/linear_model/linear_model_predict_kernel.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/infer_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/cpu/infer_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace be = dal::backend;
namespace daal_lm = daal::algorithms::linear_model;
namespace interop = dal::backend::interop;

constexpr auto daal_method = daal_lm::prediction::Method::defaultDense;

template <typename Float, daal::internal::CpuType Cpu>
using daal_lm_kernel_t = daal_lm::prediction::internal::PredictKernel<Float, daal_method, Cpu>;

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& infer,
                                           const model<Task>& m) {
    using dal::detail::check_mul_overflow;

    const auto& betas = m.get_packed_coefficients();
    const bool intp = desc.get_compute_intercept();

    const auto sample_count = infer.get_row_count();
    const auto response_count = betas.get_row_count();

    const auto feature_count = infer.get_column_count();
    ONEDAL_ASSERT((feature_count + 1) == betas.get_column_count());

    const auto resps_size = check_mul_overflow(sample_count, response_count);
    auto resps_arr = array<Float>::empty(resps_size);
    auto resps_daal_table =
        interop::convert_to_daal_homogen_table(resps_arr, sample_count, response_count);

    auto betas_daal_table = interop::convert_to_daal_table<Float>(betas);
    auto infer_daal_table = interop::convert_to_daal_table<Float>(infer);

    const auto status = dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        using cpu_t = daal::internal::CpuType;
        constexpr cpu_t cpu_type = interop::to_daal_cpu_type<decltype(cpu)>::value;
        return daal_lm_kernel_t<Float, cpu_t(cpu_type)>().compute_impl(infer_daal_table.get(),
                                                                       betas_daal_table.get(),
                                                                       resps_daal_table.get(),
                                                                       intp);
    });

    interop::status_to_exception(status);

    auto responses = homogen_table::wrap(resps_arr, sample_count, response_count);

    auto result = infer_result<Task>().set_responses(responses);

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, method::norm_eq, Task> {
    infer_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::norm_eq, task::regression>;
template struct infer_kernel_cpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
