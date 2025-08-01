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

#pragma once

#include "oneapi/dal/algo/knn/common.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"

namespace oneapi::dal::knn::backend {

template <typename Task, typename ModelImpl>
inline auto dynamic_cast_to_knn_model(const model<Task>& m) {
    const auto trained_model = dynamic_cast<ModelImpl*>(&dal::detail::get_impl(m));

    if (!trained_model) {
        throw invalid_argument{ dal::detail::error_messages::incompatible_knn_model() };
    }

    return trained_model;
}

template <typename Float>
inline auto create_daal_model_for_bf_knn(
    const daal::data_management::NumericTablePtr daal_train_data,
    const daal::data_management::NumericTablePtr daal_train_responses) {
    namespace daal_bf_knn = daal::algorithms::bf_knn_classification;
    const std::int64_t column_count = daal_train_data->getNumberOfColumns();

    const auto model_ptr = daal_bf_knn::ModelPtr(new daal_bf_knn::Model(column_count));

    // Data or responses should not be copied
    model_ptr->impl()->setData<Float>(daal_train_data, false);
    model_ptr->impl()->setLabels<Float>(daal_train_responses, false);

    return model_ptr;
}

template <typename Float, typename Task>
inline auto convert_onedal_to_daal_knn_model(const model<Task>& m) {
    namespace interop = dal::backend::interop;

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);

    // Changed to perform a copy as far as we have similar logic
    // for d4p patching; to allign performance with DAAL
    const auto daal_train_data = interop::convert_to_daal_table<Float>(trained_model->get_data());
    const auto daal_train_responses =
        interop::convert_to_daal_table<Float>(trained_model->get_responses());

    const auto model_ptr =
        create_daal_model_for_bf_knn<Float>(daal_train_data, daal_train_responses);

    return model_ptr;
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float, typename Task>
inline auto convert_onedal_to_daal_knn_model(const sycl::queue& queue, const model<Task>& m) {
    namespace interop = dal::backend::interop;

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);

    const auto daal_train_data = interop::convert_to_daal_table(queue, trained_model->get_data());
    const auto daal_train_responses =
        interop::convert_to_daal_table(queue, trained_model->get_responses());

    const auto model_ptr =
        create_daal_model_for_bf_knn<Float>(daal_train_data, daal_train_responses);

    return model_ptr;
}
#endif

} // namespace oneapi::dal::knn::backend
