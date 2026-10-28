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
#include "oneapi/dal/backend/memory.hpp"

#include "oneapi/dal/algo/svm/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"

#include "oneapi/dal/table/csr_accessor.hpp"
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

/// Copies two row blocks of a dense support-vector matrix into one freshly
/// allocated `HomogenNumericTable`: the `n_i` rows starting at `first_i`,
/// followed by the `n_j` rows starting at `first_j`.
///
/// @tparam Float      Floating-point type of the support vectors.
/// @param sv_data     Row-major aggregated support-vector matrix.
/// @param column_count Number of features per support vector.
/// @param first_i     Index of the first row of the class-i block.
/// @param n_i         Number of rows in the class-i block.
/// @param first_j     Index of the first row of the class-j block.
/// @param n_j         Number of rows in the class-j block.
/// @return A `(n_i + n_j) x column_count` dense table owning its data.
template <typename Float>
static daal::data_management::NumericTablePtr slice_pair_support_vectors_dense(
    const Float* sv_data,
    const std::int64_t column_count,
    const std::int64_t first_i,
    const std::int64_t n_i,
    const std::int64_t first_j,
    const std::int64_t n_j) {
    daal::services::Status status;
    auto pair_sv = daal::data_management::HomogenNumericTable<Float>::create(
        column_count,
        n_i + n_j,
        daal::data_management::NumericTable::doAllocate,
        &status);
    interop::status_to_exception(status);

    daal::internal::WriteOnlyRows<Float, DAAL_BASE_CPU> rows(pair_sv.get(), 0, n_i + n_j);
    interop::status_to_exception(rows.status());
    // Each class block is a contiguous row range of a row-major matrix, so one
    // copy per block is enough -- there is nothing to do per row.
    Float* dst = rows.get();
    const Float* src_i = sv_data + first_i * column_count;
    const Float* src_j = sv_data + first_j * column_count;
    dal::backend::copy(dst, src_i, n_i * column_count);
    dal::backend::copy(dst + n_i * column_count, src_j, n_j * column_count);

    return daal::data_management::NumericTablePtr{ pair_sv };
}

/// Same slice as `slice_pair_support_vectors_dense`, for a CSR support-vector
/// matrix. Column indices are copied verbatim -- both blocks share a column space.
/// CSR arrives here from a scikit-learn-intelex sparse fit, which never passes
/// through the oneDAL trainer that would have left a `model_interop_cls` behind.
///
/// @tparam Float           Floating-point type of the support-vector values.
/// @param sv_values        Non-zero values of the aggregated CSR matrix.
/// @param sv_column_indices One-based column index of every non-zero.
/// @param sv_row_offsets   One-based row offsets, `sv_row_count + 1` entries.
/// @param column_count     Number of features per support vector.
/// @param first_i          Index of the first row of the class-i block.
/// @param n_i              Number of rows in the class-i block.
/// @param first_j          Index of the first row of the class-j block.
/// @param n_j              Number of rows in the class-j block.
/// @return A `(n_i + n_j) x column_count` CSR table owning its data.
template <typename Float>
static daal::data_management::NumericTablePtr slice_pair_support_vectors_csr(
    const Float* sv_values,
    const std::int64_t* sv_column_indices,
    const std::int64_t* sv_row_offsets,
    const std::int64_t column_count,
    const std::int64_t first_i,
    const std::int64_t n_i,
    const std::int64_t first_j,
    const std::int64_t n_j) {
    const std::int64_t pair_row_count = n_i + n_j;
    const std::int64_t nnz_i = sv_row_offsets[first_i + n_i] - sv_row_offsets[first_i];
    const std::int64_t nnz_j = sv_row_offsets[first_j + n_j] - sv_row_offsets[first_j];
    const std::int64_t nnz = nnz_i + nnz_j;

    // `CSRNumericTable` rejects empty value / index arrays, which an all-zero
    // block produces. Store one explicit zero instead -- same number, same model.
    // Every pair reaching this branch is fine: each sub-model's decision value
    // then collapses to its own bias. Pinned by `svm multi-class model with
    // all-zero csr support vectors`.
    const std::int64_t stored = (nnz > 0) ? nnz : std::int64_t(1);

    auto pair_values = array<Float>::zeros(stored);
    auto pair_column_indices = array<std::int64_t>::empty(stored);
    auto pair_row_offsets = array<std::int64_t>::empty(pair_row_count + 1);

    Float* values = pair_values.get_mutable_data();
    std::int64_t* column_indices = pair_column_indices.get_mutable_data();
    std::int64_t* row_offsets = pair_row_offsets.get_mutable_data();

    if (nnz == 0) {
        column_indices[0] = 1;
        row_offsets[0] = 1;
        for (std::int64_t r = 1; r <= pair_row_count; ++r) {
            row_offsets[r] = 2;
        }
        return daal::data_management::NumericTablePtr{ interop::convert_to_daal_csr_table(
            pair_values,
            pair_column_indices,
            pair_row_offsets,
            pair_row_count,
            column_count) };
    }

    // Zero-based index of the first non-zero of each block in the source arrays.
    const std::int64_t src_i = sv_row_offsets[first_i] - 1;
    const std::int64_t src_j = sv_row_offsets[first_j] - 1;
    for (std::int64_t e = 0; e < nnz_i; ++e) {
        values[e] = sv_values[src_i + e];
        column_indices[e] = sv_column_indices[src_i + e];
    }
    for (std::int64_t e = 0; e < nnz_j; ++e) {
        values[nnz_i + e] = sv_values[src_j + e];
        column_indices[nnz_i + e] = sv_column_indices[src_j + e];
    }

    row_offsets[0] = 1;
    for (std::int64_t r = 0; r < n_i; ++r) {
        row_offsets[r + 1] = sv_row_offsets[first_i + r + 1] - sv_row_offsets[first_i] + 1;
    }
    for (std::int64_t r = 0; r < n_j; ++r) {
        row_offsets[n_i + r + 1] =
            nnz_i + sv_row_offsets[first_j + r + 1] - sv_row_offsets[first_j] + 1;
    }

    return daal::data_management::NumericTablePtr{ interop::convert_to_daal_csr_table(
        pair_values,
        pair_column_indices,
        pair_row_offsets,
        pair_row_count,
        column_count) };
}

// Rebuilds a daal multi_class_classifier::Model from the aggregated public arrays
// of a oneapi svm model. The shapes in svm/common.hpp do not pin the row order, so
// this is a contract with the multiclass CPU trainer:
//   - support-vector rows sorted by class, block sizes in n_support_per_class;
//   - for an SV of class c, the coefficient against class o is in coeffs column
//     (o - 1) when o > c and column o when o < c;
//   - pair order matches daal getClassIndices(isSvmModel=true):
//     (0,1), (0,2), ..., (0, k-1), (1,2), ..., (k-2, k-1).
// Dense and CSR support vectors are both accepted. `data_layout` is passed in only
// to be compared against the support-vector layout.
template <typename Float, typename Task>
static daal_multiclass::ModelPtr convert_to_daal_multiclass_model(
    const model<Task>& trained_model,
    const std::int64_t column_count,
    const std::uint64_t class_count,
    const daal::data_management::NumericTableIface::StorageLayout data_layout) {
    const auto sv_table = trained_model.get_support_vectors();
    const auto coeffs_table = trained_model.get_coeffs();
    const auto biases_table = trained_model.get_biases();
    const auto n_per_class_table = trained_model.get_n_support_per_class();

    if (!sv_table.has_data() || !coeffs_table.has_data() || !biases_table.has_data()) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    // n_support_per_class is a 1 x class_count row of int32 counts, indexed by
    // class below, so both dimensions have to match exactly.
    if (!n_per_class_table.has_data() || n_per_class_table.get_row_count() != std::int64_t(1) ||
        n_per_class_table.get_column_count() != static_cast<std::int64_t>(class_count)) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    // biases holds one scalar per pairwise sub-model, in one column.
    const std::int64_t expected_model_count =
        static_cast<std::int64_t>(class_count) * (static_cast<std::int64_t>(class_count) - 1) / 2;
    if (biases_table.get_row_count() != expected_model_count ||
        biases_table.get_column_count() != std::int64_t(1)) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    // One kernel serves both operands, so a mixed-layout pair is refused here
    // rather than handed to the kernel.
    const bool sv_is_csr = sv_table.get_kind() == dal::csr_table::kind();
    const bool data_is_csr =
        data_layout == daal::data_management::NumericTableIface::StorageLayout::csrArray;
    if (sv_is_csr != data_is_csr) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }
    // A sub-model is told the layout of its own support vectors, not the data's.
    const auto sv_layout = sv_is_csr
                               ? daal::data_management::NumericTableIface::StorageLayout::csrArray
                               : daal::data_management::NumericTableIface::StorageLayout::aos;

    auto n_per_class_arr = row_accessor<const std::int32_t>{ n_per_class_table }.pull();
    const auto n_per_class = n_per_class_arr.get_data();

    auto coeffs_arr = row_accessor<const Float>{ coeffs_table }.pull();
    // A trained model's biases are always `double` (a daal sub-model holds a
    // `double` scalar), but the accessor converts rather than reinterprets, so a
    // `float` table a caller set of its own reads back fine.
    auto biases_arr = row_accessor<const double>{ biases_table }.pull();

    // Dense support vectors are pulled as one row-major block; CSR ones as the
    // usual (values, column indices, row offsets) triple with one-based
    // indexing, which is what `convert_to_daal_csr_table` expects back.
    array<Float> sv_arr;
    array<Float> sv_values;
    array<std::int64_t> sv_column_indices;
    array<std::int64_t> sv_row_offsets;
    if (sv_is_csr) {
        std::tie(sv_values, sv_column_indices, sv_row_offsets) =
            csr_accessor<const Float>{ static_cast<const csr_table&>(sv_table) }.pull(
                { 0, -1 },
                sparse_indexing::one_based);
    }
    else {
        sv_arr = row_accessor<const Float>{ sv_table }.pull();
    }

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
    // Every count must be positive. Counts should sum up to the number of support vectors.
    std::vector<std::int64_t> class_offsets(class_count + 1, 0);
    for (std::uint64_t c = 0; c < class_count; ++c) {
        if (n_per_class[c] <= 0) {
            throw invalid_argument(
                dal::detail::error_messages::input_model_does_not_match_kernel_function());
        }
        class_offsets[c + 1] = class_offsets[c] + n_per_class[c];
    }
    if (class_offsets[class_count] != n_sv_total) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }

    const auto coeffs_data = coeffs_arr.get_data();
    const auto biases_data = biases_arr.get_data();
    const std::int64_t coeff_stride = static_cast<std::int64_t>(class_count) - 1;

    std::int64_t imodel = 0;
    for (std::uint64_t i = 0; i < class_count; ++i) {
        const std::int64_t n_i = n_per_class[i];
        for (std::uint64_t j = i + 1; j < class_count; ++j, ++imodel) {
            const std::int64_t n_j = n_per_class[j];
            const std::int64_t pair_n_sv = n_i + n_j;

            // Build per-pair support vectors: class-i block followed by class-j block.
            auto pair_sv_nt =
                sv_is_csr ? slice_pair_support_vectors_csr<Float>(sv_values.get_data(),
                                                                  sv_column_indices.get_data(),
                                                                  sv_row_offsets.get_data(),
                                                                  column_count,
                                                                  class_offsets[i],
                                                                  n_i,
                                                                  class_offsets[j],
                                                                  n_j)
                          : slice_pair_support_vectors_dense<Float>(sv_arr.get_data(),
                                                                    column_count,
                                                                    class_offsets[i],
                                                                    n_i,
                                                                    class_offsets[j],
                                                                    n_j);

            // Build per-pair classification coefficients (single column).
            auto pair_coeffs = daal::data_management::HomogenNumericTable<Float>::create(
                1,
                pair_n_sv,
                daal::data_management::NumericTable::doAllocate,
                &status);
            interop::status_to_exception(status);
            {
                daal::internal::WriteOnlyRows<Float, DAAL_BASE_CPU> rows(pair_coeffs.get(),
                                                                         0,
                                                                         pair_n_sv);
                interop::status_to_exception(rows.status());
                Float* dst = rows.get();
                // Class-i SVs: pairwise (i, j) with j > i -> coeff column (j - 1).
                const std::int64_t col_for_i = static_cast<std::int64_t>(j) - 1;
                for (std::int64_t r = 0; r < n_i; ++r) {
                    dst[r] = coeffs_data[(class_offsets[i] + r) * coeff_stride + col_for_i];
                }
                // Class-j SVs: pairwise (i, j) with i < j -> coeff column i.
                const std::int64_t col_for_j = static_cast<std::int64_t>(i);
                for (std::int64_t r = 0; r < n_j; ++r) {
                    dst[n_i + r] = coeffs_data[(class_offsets[j] + r) * coeff_stride + col_for_j];
                }
            }

            // Per-pair binary svm sub-model.
            auto pair_model = daal::services::SharedPtr<daal_svm::internal::ModelImpl>(
                new daal_svm::internal::ModelImpl(Float(0),
                                                  static_cast<std::size_t>(column_count),
                                                  sv_layout,
                                                  status));
            interop::status_to_exception(status);
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
        // model_interop is polymorphic and a regression model carries a
        // different derived type, so the downcast has to be checked: a
        // mismatched interop payload is a wrong-model error, not a crash.
        const auto* interop_model_cls = dynamic_cast<const model_interop_cls*>(interop_model);
        if (!interop_model_cls) {
            throw invalid_argument(
                dal::detail::error_messages::input_model_does_not_match_kernel_function());
        }
        daal_model = interop_model_cls->get_model();
    }
    else {
        // Public-setter-only model (e.g. cross-device round-trip without
        // serialization): rebuild the per-pair daal sub-models from the
        // aggregated SVs / coeffs / biases on the fly.
        daal_model = convert_to_daal_multiclass_model<Float, Task>(trained_model,
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
    // The model carries the authoritative class_count. The descriptor has to
    // agree, so a stale binary descriptor cannot route a multi-class model down
    // the binary path or the other way round.
    const std::int64_t class_count = trained_model.get_class_count();
    if (desc.get_class_count() != class_count) {
        throw invalid_argument(
            dal::detail::error_messages::input_model_does_not_match_kernel_function());
    }

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
