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

#pragma once

#include "oneapi/dal/algo/knn/common.hpp"
#include "oneapi/dal/algo/knn/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::knn {

#define KNN_SERIALIZABLE(Task, ClassificationId, RegressionId, SearchId) \
    ONEDAL_SERIALIZABLE_MAP3(Task,                                       \
                             (task::classification, ClassificationId),   \
                             (task::regression, RegressionId),           \
                             (task::search, SearchId))

template <typename Task>
class detail::v1::model_impl : public base {
public:
    model_impl() = default;

    virtual backend::model_interop* get_interop() = 0;
};

namespace backend {

template <typename Task>
using model_impl = detail::model_impl<Task>;

template <typename Task>
class brute_force_model_impl : public model_impl<Task>,
                               public KNN_SERIALIZABLE(Task,
                                                       knn_brute_force_classification_model_impl_id,
                                                       knn_brute_force_regression_model_impl_id,
                                                       knn_brute_force_search_model_impl_id) {
public:
    brute_force_model_impl() = default;

    brute_force_model_impl(const table& data, const table& responses)
            : data_(data),
              responses_(responses) {}

    backend::model_interop* get_interop() override {
        return nullptr;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(data_, responses_);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(data_, responses_);
    }

    table get_data() {
        return data_;
    }

    table get_responses() {
        return responses_;
    }

private:
    table data_;
    table responses_;
};

template <typename Task>
class kd_tree_model_impl : public model_impl<Task>,
                           public KNN_SERIALIZABLE(Task,
                                                   knn_kd_tree_classification_model_impl_id,
                                                   knn_kd_tree_regression_model_impl_id,
                                                   knn_kd_tree_search_model_impl_id) {
public:
    kd_tree_model_impl() : interop_(nullptr) {}
    kd_tree_model_impl(backend::model_interop* interop) : interop_(interop) {}
    kd_tree_model_impl(const kd_tree_model_impl&) = delete;
    kd_tree_model_impl& operator=(const kd_tree_model_impl&) = delete;

    kd_tree_model_impl(kd_tree_model_impl&& other) noexcept = default;

    kd_tree_model_impl& operator=(kd_tree_model_impl&& other) noexcept = default;

    ~kd_tree_model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() override {
        return interop_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        delete interop_;
        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

private:
    backend::model_interop* interop_;
};

} // namespace backend
} // namespace oneapi::dal::knn
