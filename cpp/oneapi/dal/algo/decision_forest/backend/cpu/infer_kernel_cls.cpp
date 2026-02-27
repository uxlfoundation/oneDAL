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
#include <daal/src/algorithms/dtrees/forest/df_hyperparameter_impl.h>
#include <daal/src/algorithms/dtrees/forest/classification/df_classification_predict_dense_default_batch.h>

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/algo/decision_forest/backend/cpu/infer_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_cpu;
using model_t = model<task::classification>;
using input_t = infer_input<task::classification>;
using result_t = infer_result<task::classification>;
using param_t = detail::infer_parameters<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_df = daal::algorithms::decision_forest;

using daal_hyperparameters_t = daal_df::prediction::internal::Hyperparameter;

namespace daal_df = daal::algorithms::decision_forest;
namespace daal_df_cls_pred = daal_df::classification::prediction;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using cls_dense_predict_kernel_t =
    daal_df_cls_pred::internal::PredictKernel<Float, daal_df_cls_pred::defaultDense, Cpu>;

static daal_df::classification::ModelPtr get_daal_model(const model_t& trained_model) {
    const model_interop* interop_model = dal::detail::get_impl(trained_model).get_interop();
    if (!interop_model) {
        throw dal::internal_error(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    return static_cast<const model_interop_cls*>(interop_model)->get_model();
}

static daal_hyperparameters_t convert_parameters(const param_t& params) {
    using daal_df::prediction::internal::HyperparameterId;
    using daal_df::prediction::internal::DoubleHyperparameterId;

    const std::int64_t block = params.get_block_size();
    const std::int64_t minTrees = params.get_min_trees_for_threading();
    const std::int64_t minRows = params.get_min_number_of_rows_for_vect_seq_compute();
    const double scale = params.get_scale_factor_for_vect_parallel_compute();

    daal_hyperparameters_t daal_hyperparameter;

    auto status = daal_hyperparameter.set(HyperparameterId::blockSize, block);
    status |= daal_hyperparameter.set(HyperparameterId::minTreesForThreading, minTrees);
    status |= daal_hyperparameter.set(HyperparameterId::minNumberOfRowsForVectSeqCompute, minRows);
    status |=
        daal_hyperparameter.set(DoubleHyperparameterId::scaleFactorForVectParallelCompute, scale);

    interop::status_to_exception(status);

    return daal_hyperparameter;
}

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const param_t& params,
                                 const model_t& trained_model,
                                 const table& data) {
    const std::int64_t row_count = data.get_row_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto daal_model = get_daal_model(trained_model);

    auto daal_input = daal::algorithms::classifier::prediction::Input();
    daal_input.set(daal::algorithms::classifier::prediction::data, daal_data);
    daal_input.set(daal::algorithms::classifier::prediction::model, daal_model);

    const auto daal_voting_mode = convert_to_daal_voting_mode(desc.get_voting_mode());
    const auto daal_parameter =
        daal_df_cls_pred::Parameter(dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
                                    daal_voting_mode);

    daal::data_management::NumericTablePtr daal_responses_res;
    daal::data_management::NumericTablePtr daal_responses_prob_res;
    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_responses)) {
        daal_responses_res = interop::allocate_daal_homogen_table<Float>(row_count, 1);
    }
    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
        daal_responses_prob_res =
            interop::allocate_daal_homogen_table<Float>(row_count, desc.get_class_count());
    }

    const daal_df::classification::Model* const daal_model_ptr = daal_model.get();
    const daal_hyperparameters_t& hyperparameters = convert_parameters(params);
    interop::status_to_exception(interop::call_daal_kernel<Float, cls_dense_predict_kernel_t>(
        ctx,
        daal::services::internal::hostApp(daal_input),
        daal_data.get(),
        daal_model_ptr,
        daal_responses_res.get(),
        daal_responses_prob_res.get(),
        desc.get_class_count(),
        daal_voting_mode,
        &hyperparameters));

    result_t res;

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_responses)) {
        auto table_class_responses =
            interop::convert_from_daal_homogen_table<Float>(daal_responses_res);
        res.set_responses(table_class_responses);
    }

    if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
        auto table_class_probs =
            interop::convert_from_daal_homogen_table<Float>(daal_responses_prob_res);
        res.set_probabilities(table_class_probs);
    }

    return res;
}

template <typename Float>
static result_t infer(const context_cpu& ctx,
                      const descriptor_t& desc,
                      const param_t& params,
                      const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, params, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_cpu<Float, method::by_default, task::classification> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const param_t& params,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, params, input);
    }
};

template struct infer_kernel_cpu<float, method::by_default, task::classification>;
template struct infer_kernel_cpu<double, method::by_default, task::classification>;

} // namespace oneapi::dal::decision_forest::backend
