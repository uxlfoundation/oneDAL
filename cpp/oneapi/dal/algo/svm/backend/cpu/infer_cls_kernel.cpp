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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/svm/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/svm/svm_predict_kernel.h>
#include <daal/src/algorithms/multiclassclassifier/multiclassclassifier_train_kernel.h>
#include <daal/src/algorithms/multiclassclassifier/multiclassclassifier_predict_kernel.h>

#include "algorithms/svm/svm_predict.h"

namespace oneapi::dal::svm::backend {

using dal::backend::context_cpu;

namespace daal_svm = daal::algorithms::svm;
namespace daal_classifier = daal::algorithms::classifier;
namespace daal_multiclass = daal::algorithms::multi_class_classifier;

namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_svm_predict_kernel_t =
    daal_svm::prediction::internal::SVMPredictImpl<daal_svm::prediction::defaultDense, Float, Cpu>;

template <typename Float, daal::CpuType Cpu>
using daal_multiclass_kernel_t =
    daal_multiclass::prediction::internal::MultiClassClassifierPredictKernel<
        daal_multiclass::prediction::voteBased,
        daal_multiclass::training::oneAgainstOne,
        Float,
        Cpu>;

template <typename Float, typename Task>
static infer_result<Task> call_multiclass_daal_kernel(const context_cpu& ctx,
                                                      const detail::descriptor_base<Task>& desc,
                                                      const model<Task>& trained_model,
                                                      const table& data,
                                                      const daal_svm::Parameter& daal_parameter,
                                                      const std::uint64_t class_count) {
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t row_count = data.get_row_count();
    auto arr_response = array<Float>::empty(row_count * 1);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_support_vectors =
        interop::convert_to_daal_table<Float>(trained_model.get_support_vectors());
    const auto daal_coeffs = interop::convert_to_daal_table<Float>(trained_model.get_coeffs());
    const auto daal_biases = interop::convert_to_daal_table<double>(trained_model.get_biases());
    const auto daal_iterations = interop::convert_to_daal_table<int>(trained_model.get_iteration_counts());

    auto daal_model = daal_model_builder{}
                          .set_support_vectors(daal_support_vectors)
                          .set_coeffs(daal_coeffs)
                          .set_biases(daal_biases)
                          .set_iteration_counts(daal_iterations);

    const std::int64_t model_count = class_count * (class_count - 1) / 2;
    using svm_batch_t = typename daal_svm::prediction::Batch<Float>;

    daal_multiclass::Parameter daal_multiclass_parameter(class_count);
    auto svm_batch = daal::services::SharedPtr<svm_batch_t>(new svm_batch_t());
    svm_batch->parameter = daal_parameter;
    daal_multiclass_parameter.prediction =
        daal::services::staticPointerCast<daal_classifier::prediction::Batch>(svm_batch);

    const auto daal_response = interop::convert_to_daal_homogen_table(arr_response, row_count, 1);

    auto arr_decision_function = array<Float>::empty(row_count * model_count);
    const auto daal_decision_function =
        interop::convert_to_daal_homogen_table(arr_decision_function, row_count, model_count);

    const auto daal_layout = daal_data->getDataLayout();
    daal::services::Status status;
    auto daal_svm_model_ptr = new daal_svm::internal::ModelImpl(Float(0), class_count, column_count, daal_layout, status);
    daal_svm::ModelPtr daal_svm_model(daal_svm_model_ptr);
    interop::status_to_exception(status);
    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_multiclass_kernel_t>(ctx,
                                                                   daal_data.get(),
                                                                   &daal_model,
                                                                   daal_svm_model_ptr,
                                                                   daal_response.get(),
                                                                   daal_decision_function.get(),
                                                                   &daal_multiclass_parameter));

    return infer_result<Task>()
        .set_decision_function(dal::detail::homogen_table_builder{}
                                   .reset(arr_decision_function, row_count, model_count)
                                   .build())
        .set_responses(
            dal::detail::homogen_table_builder{}.reset(arr_response, row_count, 1).build());
}

template <typename Float, typename Task>
static infer_result<Task> call_binary_daal_kernel(const context_cpu& ctx,
                                                  const detail::descriptor_base<Task>& desc,
                                                  const model<Task>& trained_model,
                                                  const table& data,
                                                  const daal_svm::Parameter daal_parameter) {
    const std::int64_t row_count = data.get_row_count();
    auto arr_response = array<Float>::empty(row_count * 1);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_support_vectors =
        interop::convert_to_daal_table<Float>(trained_model.get_support_vectors());
    const auto daal_coeffs = interop::convert_to_daal_table<Float>(trained_model.get_coeffs());

    const auto daal_biases = interop::convert_to_daal_table<double>(trained_model.get_biases());

    auto daal_model = daal_model_builder{}
                          .set_support_vectors(daal_support_vectors)
                          .set_coeffs(daal_coeffs)
                          .set_biases(daal_biases);

    auto arr_decision_function = array<Float>::empty(row_count * 1);
    const auto daal_decision_function =
        interop::convert_to_daal_homogen_table(arr_decision_function, row_count, 1);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_svm_predict_kernel_t>(ctx,
                                                                    daal_data,
                                                                    &daal_model,
                                                                    *daal_decision_function,
                                                                    &daal_parameter));

    auto response_data = arr_response.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; ++i) {
        response_data[i] = arr_decision_function[i] >= 0 ? trained_model.get_second_class_response()
                                                         : trained_model.get_first_class_response();
    }

    return infer_result<Task>()
        .set_decision_function(
            dal::detail::homogen_table_builder{}.reset(arr_decision_function, row_count, 1).build())
        .set_responses(
            dal::detail::homogen_table_builder{}.reset(arr_response, row_count, 1).build());
}

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const model<Task>& trained_model,
                                           const table& data) {
    const std::int64_t class_count = dal::detail::get_impl(trained_model).class_count;

    auto kernel_impl = detail::get_kernel_function_impl(desc);
    if (!kernel_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }
    const bool is_dense{ data.get_kind() != dal::csr_table::kind() };
    const auto daal_kernel = kernel_impl->get_daal_kernel_function(is_dense);
    daal_svm::Parameter daal_parameter(daal_kernel);

    if (class_count > 2) {
        return call_multiclass_daal_kernel<Float, Task>(ctx,
                                                        desc,
                                                        trained_model,
                                                        data,
                                                        daal_parameter,
                                                        class_count);
    }
    else {
        return call_binary_daal_kernel<Float, Task>(ctx, desc, trained_model, data, daal_parameter);
    }
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, method::by_default, Task> {
    infer_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::by_default, task::classification>;
template struct infer_kernel_cpu<double, method::by_default, task::classification>;
template struct infer_kernel_cpu<float, method::by_default, task::nu_classification>;
template struct infer_kernel_cpu<double, method::by_default, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
