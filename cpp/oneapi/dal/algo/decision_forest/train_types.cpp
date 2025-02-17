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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

namespace detail::v1 {

template <>
struct train_parameters_impl<task::classification> : public base {
    std::int64_t small_classes_threshold = 8l;
    std::int64_t min_part_coefficient = 4l;
    std::int64_t min_size_coefficient = 24000l;
};

train_parameters<task::classification>::train_parameters() : impl_(new train_parameters_impl<task::classification>{}) {}

std::int64_t train_parameters<task::classification>::get_small_classes_threshold() const {
    return impl_->small_classes_threshold;
}

void train_parameters<task::classification>::set_small_classes_threshold_impl(std::int64_t val) {
    impl_->small_classes_threshold = val;
}

std::int64_t train_parameters<task::classification>::get_min_part_coefficient() const {
    return impl_->min_part_coefficient;
}

void train_parameters<task::classification>::set_min_part_coefficient_impl(std::int64_t val) {
    impl_->min_part_coefficient = val;
}

std::int64_t train_parameters<task::classification>::get_min_size_coefficient() const {
    return impl_->min_size_coefficient;
}

void train_parameters<task::classification>::set_min_size_coefficient_impl(std::int64_t val) {
    impl_->min_size_coefficient = val;
}

template <>
struct train_parameters_impl<task::regression> : public base {};

train_parameters<task::regression>::train_parameters() : impl_(new train_parameters_impl<task::regression>{}) {}

// template class ONEDAL_EXPORT train_parameters<task::classification>;
// template class ONEDAL_EXPORT train_parameters<task::regression>;

} // namespace detail::v1

template <typename Task>
class detail::v1::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& responses, const table& weights)
            : data(data),
              responses(responses),
              weights(weights) {}

    table data;
    table responses;
    table weights;
};

template <typename Task>
class detail::v1::train_result_impl : public base {
public:
    model<Task> trained_model;

    table oob_err;
    table oob_err_per_observation;
    table oob_err_accuracy;
    table oob_err_r2;
    table oob_err_decision_function;
    table oob_err_prediction;
    table variable_importance;
};


using detail::v1::train_parameters;
using detail::v1::train_input_impl;
using detail::v1::train_result_impl;

namespace v1 {

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& train_result<Task>::get_oob_err() const {
    return impl_->oob_err;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_per_observation() const {
    return impl_->oob_err_per_observation;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_accuracy() const {
    return impl_->oob_err_accuracy;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_r2() const {
    return impl_->oob_err_r2;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_decision_function() const {
    return impl_->oob_err_decision_function;
}

template <typename Task>
const table& train_result<Task>::get_oob_err_prediction() const {
    return impl_->oob_err_prediction;
}

template <typename Task>
const table& train_result<Task>::get_var_importance() const {
    return impl_->variable_importance;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_impl(const table& value) {
    impl_->oob_err = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_per_observation_impl(const table& value) {
    impl_->oob_err_per_observation = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_accuracy_impl(const table& value) {
    impl_->oob_err_accuracy = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_r2_impl(const table& value) {
    impl_->oob_err_r2 = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_decision_function_impl(const table& value) {
    impl_->oob_err_decision_function = value;
}

template <typename Task>
void train_result<Task>::set_oob_err_prediction_impl(const table& value) {
    impl_->oob_err_prediction = value;
}

template <typename Task>
void train_result<Task>::set_var_importance_impl(const table& value) {
    impl_->variable_importance = value;
}
template class ONEDAL_EXPORT train_result<task::classification>;
template class ONEDAL_EXPORT train_result<task::regression>;

} // namespace v1

namespace v2 {

template <typename Task>
train_input<Task>::train_input(const table& data, const table& responses, const table& weights)
        : impl_(new train_input_impl<Task>(data, responses, weights)) {}

template <typename Task>
const table& train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& train_input<Task>::get_responses() const {
    return impl_->responses;
}

template <typename Task>
const table& train_input<Task>::get_weights() const {
    return impl_->weights;
}

template <typename Task>
void train_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void train_input<Task>::set_responses_impl(const table& value) {
    impl_->responses = value;
}

template <typename Task>
void train_input<Task>::set_weights_impl(const table& value) {
    impl_->weights = value;
}

template class ONEDAL_EXPORT train_input<task::classification>;
template class ONEDAL_EXPORT train_input<task::regression>;

} // namespace v2
} // namespace oneapi::dal::decision_forest
