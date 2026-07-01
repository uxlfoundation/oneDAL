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

#include <vector>

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

template <typename Float, daal::internal::CpuType Cpu>
using daal_svm_predict_kernel_t =
    daal_svm::prediction::internal::SVMPredictImpl<daal_svm::prediction::defaultDense, Float, Cpu>;

template <typename Float, daal::internal::CpuType Cpu>
using daal_multiclass_kernel_t =
    daal_multiclass::prediction::internal::MultiClassClassifierPredictKernel<
        daal_multiclass::prediction::voteBased,
        daal_multiclass::training::oneAgainstOne,
        Float,
        Cpu>;

// Reconstructs a daal multi_class_classifier::Model from the aggregated public
// arrays of a oneapi svm model. SVs and coeffs are stored grouped by class in
// increasing class order; n_support_per_class gives the per-class block sizes.
// Layout assumptions follow the daal one-vs-one trainer:
//   support_vectors : [sum(n_support_per_class), n_features]
//   coeffs          : [sum(n_support_per_class), class_count - 1]
//   biases          : [class_count*(class_count-1)/2, 1]
// For SV row of class c, coeff column for the duel against class o is:
//   o > c -> column (o - 1);  o < c -> column o.
// Pair iteration order matches getClassIndices(isSvmModel=true):
//   (0,1), (0,2), ..., (0,k-1), (1,2), ..., (k-2,k-1).
template <typename Float, typename Task>
static daal_multiclass::ModelPtr build_daal_multiclass_model_from_public(
    const model<Task>& trained_model,
    const std::int64_t column_count,
    const std::uint64_t class_count,
    const daal::data_management::NumericTableIface::StorageLayout daal_layout) {
    const auto sv_table = trained_model.get_support_vectors();
    const auto coeffs_table = trained_model.get_coeffs();
    const auto biases_table = trained_model.get_biases();
    const auto n_per_class_table = trained_model.get_n_support_per_class();

    if (!sv_table.has_data() || !coeffs_table.has_data() || !biases_table.has_data()) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    if (!n_per_class_table.has_data() ||
        n_per_class_table.get_column_count() != static_cast<std::int64_t>(class_count)) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }

    auto n_per_class_arr = row_accessor<const std::int32_t>{ n_per_class_table }.pull();
    const auto n_per_class = n_per_class_arr.get_data();

    auto sv_arr = row_accessor<const Float>{ sv_table }.pull();
    auto coeffs_arr = row_accessor<const Float>{ coeffs_table }.pull();
    auto biases_arr = row_accessor<const double>{ biases_table }.pull();

    const std::int64_t n_sv_total = sv_table.get_row_count();
    if (coeffs_table.get_row_count() != n_sv_total ||
        coeffs_table.get_column_count() != static_cast<std::int64_t>(class_count) - 1) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }

    daal::services::Status status;
    daal_multiclass::Parameter daal_par(class_count);
    auto multiclass_model = daal_multiclass::Model::create(column_count, &daal_par, &status);
    interop::status_to_exception(status);

    // Cumulative per-class offsets into the aggregated SV / coeff matrices.
    std::vector<std::int64_t> class_offsets(class_count + 1, 0);
    for (std::uint64_t c = 0; c < class_count; ++c) {
        class_offsets[c + 1] = class_offsets[c] + n_per_class[c];
    }
    if (class_offsets[class_count] != n_sv_total) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }

    const auto sv_data = sv_arr.get_data();
    const auto coeffs_data = coeffs_arr.get_data();
    const auto biases_data = biases_arr.get_data();
    const std::int64_t coeff_stride = static_cast<std::int64_t>(class_count) - 1;

    std::int64_t imodel = 0;
    for (std::uint64_t i = 0; i < class_count; ++i) {
        const std::int64_t n_i = n_per_class[i];
        for (std::uint64_t j = i + 1; j < class_count; ++j, ++imodel) {
            const std::int64_t n_j = n_per_class[j];
            const std::int64_t pair_n_sv = n_i + n_j;
            if (pair_n_sv == 0) {
                multiclass_model->setTwoClassClassifierModel(
                    imodel,
                    daal::algorithms::classifier::ModelPtr{});
                continue;
            }

            // Build per-pair support vectors: class-i block followed by class-j block.
            auto pair_sv = daal::data_management::HomogenNumericTable<Float>::create(
                column_count,
                pair_n_sv,
                daal::data_management::NumericTable::doAllocate,
                &status);
            interop::status_to_exception(status);
            {
                daal::data_management::BlockDescriptor<Float> blk;
                pair_sv->getBlockOfRows(0, pair_n_sv, daal::data_management::writeOnly, blk);
                Float* dst = blk.getBlockPtr();
                const Float* src_i = sv_data + class_offsets[i] * column_count;
                const Float* src_j = sv_data + class_offsets[j] * column_count;
                for (std::int64_t r = 0; r < n_i; ++r) {
                    for (std::int64_t k = 0; k < column_count; ++k) {
                        dst[r * column_count + k] = src_i[r * column_count + k];
                    }
                }
                for (std::int64_t r = 0; r < n_j; ++r) {
                    for (std::int64_t k = 0; k < column_count; ++k) {
                        dst[(n_i + r) * column_count + k] = src_j[r * column_count + k];
                    }
                }
                pair_sv->releaseBlockOfRows(blk);
            }

            // Build per-pair classification coefficients (single column).
            auto pair_coeffs = daal::data_management::HomogenNumericTable<Float>::create(
                1,
                pair_n_sv,
                daal::data_management::NumericTable::doAllocate,
                &status);
            interop::status_to_exception(status);
            {
                daal::data_management::BlockDescriptor<Float> blk;
                pair_coeffs->getBlockOfRows(0, pair_n_sv, daal::data_management::writeOnly, blk);
                Float* dst = blk.getBlockPtr();
                // Class-i SVs: duel against j (j > i) -> coeff column (j - 1).
                const std::int64_t col_for_i = static_cast<std::int64_t>(j) - 1;
                for (std::int64_t r = 0; r < n_i; ++r) {
                    dst[r] = coeffs_data[(class_offsets[i] + r) * coeff_stride + col_for_i];
                }
                // Class-j SVs: duel against i (i < j) -> coeff column i.
                const std::int64_t col_for_j = static_cast<std::int64_t>(i);
                for (std::int64_t r = 0; r < n_j; ++r) {
                    dst[n_i + r] = coeffs_data[(class_offsets[j] + r) * coeff_stride + col_for_j];
                }
                pair_coeffs->releaseBlockOfRows(blk);
            }

            // Per-pair binary svm sub-model.
            auto pair_model = daal::services::SharedPtr<daal_svm::internal::ModelImpl>(
                new daal_svm::internal::ModelImpl(Float(0),
                                                  static_cast<std::size_t>(column_count),
                                                  daal_layout,
                                                  status));
            interop::status_to_exception(status);
            daal::data_management::NumericTablePtr pair_sv_nt = pair_sv;
            daal::data_management::NumericTablePtr pair_coeffs_nt = pair_coeffs;
            pair_model->setSupportVectors(pair_sv_nt);
            pair_model->setClassificationCoefficients(pair_coeffs_nt);
            pair_model->setBias(static_cast<double>(biases_data[imodel]));

            multiclass_model->setTwoClassClassifierModel(
                imodel,
                daal::services::staticPointerCast<daal::algorithms::classifier::Model>(pair_model));
        }
    }

    return multiclass_model;
}

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
    const auto daal_layout = daal_data->getDataLayout();
    const model_interop* interop_model = dal::detail::get_impl(trained_model).get_interop();
    daal_multiclass::ModelPtr daal_model;
    if (interop_model) {
        daal_model = static_cast<const model_interop_cls*>(interop_model)->get_model();
    }
    else {
        // Public-setter-only model (e.g. cross-device round-trip without
        // serialization): rebuild the per-pair daal sub-models from the
        // aggregated SVs / coeffs / biases on the fly.
        daal_model = build_daal_multiclass_model_from_public<Float, Task>(trained_model,
                                                                          column_count,
                                                                          class_count,
                                                                          daal_layout);
    }
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

    daal::services::Status status;
    auto daal_svm_model_ptr =
        new daal_svm::internal::ModelImpl(Float(0), class_count, column_count, daal_layout, status);
    daal_svm::ModelPtr daal_svm_model(daal_svm_model_ptr);
    interop::status_to_exception(status);
    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_multiclass_kernel_t>(ctx,
                                                                   daal_data.get(),
                                                                   daal_model.get(),
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
    const std::int64_t class_count = desc.get_class_count();

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
