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

#include "oneapi/dal/algo/knn/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::knn {

template <typename Task>
class detail::v1::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& responses = table{})
            : data(data),
              responses(responses) {}

    table data;
    table responses;
};

template <typename Task>
class detail::v1::train_result_impl : public base {
public:
    model<Task> trained_model;
};

using detail::v1::train_input_impl;
using detail::v1::train_result_impl;

namespace v1 {

template <typename Task>
train_input<Task>::train_input(const table& data, const table& responses)
        : impl_(new train_input_impl<Task>(data, responses)) {}

template <typename Task>
train_input<Task>::train_input(const table& data) : impl_(new train_input_impl<Task>(data)) {}

template <typename Task>
train_input<Task>::~train_input() {}

template <typename Task>
train_input<Task>::train_input(const train_input& other)
        : impl_(new train_input_impl<Task>(*other.impl_)) {}

template <typename Task>
train_input<Task>::train_input(train_input&& other) noexcept : impl_(std::move(other.impl_)) {}

template <typename Task>
train_input<Task>& train_input<Task>::operator=(const train_input& other) {
    if (this != &other) {
        train_input<Task> tmp(other);
        swap(*this, tmp);
    }
    return *this;
}

template <typename Task>
train_input<Task>& train_input<Task>::operator=(train_input&& other) noexcept {
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

template <typename Task>
const table& train_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& train_input<Task>::get_responses() const {
    return impl_->responses;
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
void train_input<Task>::swap(train_input<Task>& a, train_input<Task>& b) noexcept {
    std::swap(a.impl_, b.impl_);
}

template <typename Task>
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template class ONEDAL_EXPORT train_input<task::classification>;
template class ONEDAL_EXPORT train_result<task::classification>;
template class ONEDAL_EXPORT train_input<task::regression>;
template class ONEDAL_EXPORT train_result<task::regression>;
template class ONEDAL_EXPORT train_input<task::search>;
template class ONEDAL_EXPORT train_result<task::search>;

} // namespace v1
} // namespace oneapi::dal::knn
