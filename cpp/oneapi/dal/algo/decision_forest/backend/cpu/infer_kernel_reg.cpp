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

#include <daal/src/services/service_algo_utils.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_predict_dense_default_batch.h>

#include "oneapi/dal/algo/decision_forest/backend/cpu/infer_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_cpu;
using model_t = model<task::regression>;
using input_t = infer_input<task::regression>;
using result_t = infer_result<task::regression>;
using param_t = detail::infer_parameters<task::regression>;
using descriptor_t = detail::descriptor_base<task::regression>;

namespace daal_df = daal::algorithms::decision_forest;
namespace daal_df_reg_pred = daal_df::regression::prediction;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using reg_dense_predict_kernel_t =
    daal_df_reg_pred::internal::PredictKernel<Float, daal_df_reg_pred::defaultDense, Cpu>;

static daal_df::regression::ModelPtr get_daal_model(const model_t& trained_model) {
    const model_interop* interop_model = dal::detail::get_impl(trained_model).get_interop();
    if (!interop_model) {
        throw dal::internal_error(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    return static_cast<const model_interop_reg*>(interop_model)->get_model();
}

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const model_t& trained_model,
                                 const table& data) {
    const std::int64_t row_count = data.get_row_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto daal_model = get_daal_model(trained_model);

    auto daal_input = daal_df_reg_pred::Input();
    daal_input.set(daal_df_reg_pred::data, daal_data);
    daal_input.set(daal_df_reg_pred::model, daal_model);

    daal::data_management::NumericTablePtr daal_responses_res =
        interop::allocate_daal_homogen_table<Float>(row_count, 1);

    const daal_df::regression::Model* const daal_model_ptr =
        static_cast<daal_df::regression::Model*>(daal_model.get());
    interop::status_to_exception(interop::call_daal_kernel<Float, reg_dense_predict_kernel_t>(
        ctx,
        daal::services::internal::hostApp(daal_input),
        daal_data.get(),
        daal_model_ptr,
        daal_responses_res.get()));

    return result_t{}.set_responses(
        interop::convert_from_daal_homogen_table<Float>(daal_responses_res));
}

template <typename Float>
static result_t infer(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_cpu<Float, method::by_default, task::regression> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const param_t& params,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::by_default, task::regression>;
template struct infer_kernel_cpu<double, method::by_default, task::regression>;

} // namespace oneapi::dal::decision_forest::backend
