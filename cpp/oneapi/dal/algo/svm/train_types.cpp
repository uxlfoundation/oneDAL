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

#include "oneapi/dal/algo/svm/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/svm/backend/model_impl.hpp"

namespace oneapi::dal::svm {

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
    table support_indices;
};

using detail::v1::train_input_impl;
using detail::v1::train_result_impl;

namespace v1 {

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

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& train_result<Task>::get_support_vectors() const {
    return impl_->trained_model.get_support_vectors();
}

template <typename Task>
const table& train_result<Task>::get_support_indices() const {
    return impl_->support_indices;
}

template <typename Task>
const table& train_result<Task>::get_coeffs() const {
    return impl_->trained_model.get_coeffs();
}

template <typename Task>
double train_result<Task>::get_bias() const {
    return dal::detail::get_impl(impl_->trained_model).bias;
}

template <typename Task>
const table& train_result<Task>::get_biases() const {
    return impl_->trained_model.get_biases();
}

template <typename Task>
std::int64_t train_result<Task>::get_support_vector_count() const {
    return impl_->trained_model.get_support_vector_count();
}

template <typename Task>
const table& train_result<Task>::get_iteration_counts() const {
    return impl_->trained_model.get_iteration_counts();
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void train_result<Task>::set_support_vectors_impl(const table& value) {
    impl_->trained_model.set_support_vectors(value);
}

template <typename Task>
void train_result<Task>::set_support_indices_impl(const table& value) {
    impl_->support_indices = value;
}

template <typename Task>
void train_result<Task>::set_coeffs_impl(const table& value) {
    impl_->trained_model.set_coeffs(value);
}

template <typename Task>
void train_result<Task>::set_bias_impl(double value) {
    dal::detail::get_impl(impl_->trained_model).bias = value;
}

template <typename Task>
void train_result<Task>::set_biases_impl(const table& value) {
    impl_->trained_model.set_biases(value);
}

template <typename Task>
void train_result<Task>::set_iteration_counts_impl(const table& value) {
    impl_->trained_model.set_iteration_counts(value);
}

template class ONEDAL_EXPORT train_input<task::classification>;
template class ONEDAL_EXPORT train_result<task::classification>;
template class ONEDAL_EXPORT train_input<task::nu_classification>;
template class ONEDAL_EXPORT train_result<task::nu_classification>;
template class ONEDAL_EXPORT train_input<task::regression>;
template class ONEDAL_EXPORT train_result<task::regression>;
template class ONEDAL_EXPORT train_input<task::nu_regression>;
template class ONEDAL_EXPORT train_result<task::nu_regression>;

} // namespace v1
} // namespace oneapi::dal::svm
