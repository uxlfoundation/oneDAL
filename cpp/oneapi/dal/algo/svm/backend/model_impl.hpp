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

#include "oneapi/dal/algo/svm/common.hpp"
#include "oneapi/dal/algo/svm/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::svm {

#define SVM_SERIALIZABLE(Task, ClassificationId, RegressionId, NuClassificationId, NuRegressionId) \
    ONEDAL_SERIALIZABLE_MAP4(Task,                                                                 \
                             (task::classification, ClassificationId),                             \
                             (task::regression, RegressionId),                                     \
                             (task::nu_classification, NuClassificationId),                        \
                             (task::nu_regression, NuRegressionId))

template <typename Task>
class detail::v1::model_impl : public SVM_SERIALIZABLE(Task,
                                                       svm_classification_model_impl_id,
                                                       svm_regression_model_impl_id,
                                                       svm_nu_classification_model_impl_id,
                                                       svm_nu_regression_model_impl_id) {
public:
    table support_vectors;
    table coeffs;
    double bias;
    table biases;
    double first_class_response;
    double second_class_response;
    std::int64_t class_count = 2;
    table iteration_counts;

    model_impl() = default;
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;
    model_impl(backend::model_interop* interop) : interop_(interop) {}

    ~model_impl() {
        delete interop_;
    }

    backend::model_interop* get_interop() const {
        return interop_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(support_vectors, coeffs, bias, biases, iteration_counts);

        if constexpr (std::is_same_v<Task, task::classification> ||
                      std::is_same_v<Task, task::nu_classification>) {
            ar(first_class_response, second_class_response, class_count);
        }

        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(support_vectors, coeffs, bias, biases, iteration_counts);

        if constexpr (std::is_same_v<Task, task::classification> ||
                      std::is_same_v<Task, task::nu_classification>) {
            ar(first_class_response, second_class_response, class_count);
        }

        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

private:
    backend::model_interop* interop_ = nullptr;
};

namespace backend {

using model_impl_cls = detail::model_impl<task::classification>;
using model_impl_nu_cls = detail::model_impl<task::nu_classification>;
using model_impl_reg = detail::model_impl<task::regression>;
using model_impl_nu_reg = detail::model_impl<task::nu_regression>;

} // namespace backend
} // namespace oneapi::dal::svm
