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

#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include <iostream>

namespace oneapi::dal::linear_regression {

namespace detail::v1 {

template <typename Task>
class train_input_impl : public base {
public:
    train_input_impl() : data(table()){};
    train_input_impl(const table& data, const table& responses = table{})
            : data(data),
              responses(responses) {}

    table data;
    table responses;
};

template <typename Task>
class train_result_impl : public base {
public:
    table intercept;
    table coefficients;

    result_option_id options;

    model<Task> trained_model;
};

template <typename Task>
class partial_train_result_impl : public base {
public:
    table xtx;
    table xty;
};

template <typename Task>
struct train_parameters_impl : public base {
    std::int64_t cpu_macro_block = 8'192l;
    std::int64_t gpu_macro_block = 16'384l;
};

template <typename Task>
train_parameters<Task>::train_parameters() : impl_(new train_parameters_impl<Task>{}) {}

template <typename Task>
std::int64_t train_parameters<Task>::get_cpu_macro_block() const {
    return impl_->cpu_macro_block;
}

template <typename Task>
void train_parameters<Task>::set_cpu_macro_block_impl(std::int64_t val) {
    impl_->cpu_macro_block = 1024;
}

template <typename Task>
std::int64_t train_parameters<Task>::get_gpu_macro_block() const {
    return impl_->gpu_macro_block;
}

template <typename Task>
void train_parameters<Task>::set_gpu_macro_block_impl(std::int64_t val) {
    impl_->gpu_macro_block = val;
}

template class ONEDAL_EXPORT train_parameters<task::regression>;

} // namespace detail::v1

using detail::v1::train_input_impl;
using detail::v1::train_result_impl;
using detail::v1::train_parameters;
using detail::v1::partial_train_result_impl;

namespace v1 {

template <typename Task>
train_input<Task>::train_input() : impl_(new train_input_impl<Task>{}) {}

template <typename Task>
train_input<Task>::train_input(const table& data, const table& responses)
        : impl_(new train_input_impl<Task>(data, responses)) {}

template <typename Task>
train_input<Task>::train_input(const table& data) : impl_(new train_input_impl<Task>(data)) {}

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
train_result<Task>::train_result() : impl_(new train_result_impl<Task>{}) {}

template <typename Task>
const model<Task>& train_result<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
void train_result<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
const table& train_result<Task>::get_intercept() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::intercept)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->intercept;
}

template <typename Task>
void train_result<Task>::set_intercept_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::intercept)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->intercept = value;
}

template <typename Task>
const table& train_result<Task>::get_coefficients() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::coefficients)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->coefficients;
}

template <typename Task>
void train_result<Task>::set_coefficients_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::coefficients)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->coefficients = value;
}

template <typename Task>
const table& train_result<Task>::get_packed_coefficients() const {
    return impl_->trained_model.get_packed_coefficients();
}

template <typename Task>
void train_result<Task>::set_packed_coefficients_impl(const table& value) {
    impl_->trained_model.set_packed_coefficients(value);
}

template <typename Task>
const result_option_id& train_result<Task>::get_result_options() const {
    return impl_->options;
}

template <typename Task>
void train_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->options = value;
}

template <typename Task>
partial_train_input<Task>::partial_train_input(const table& data, const table& responses)
        : train_input<Task>(data, responses),
          prev_() {}

template <typename Task>
partial_train_input<Task>::partial_train_input(const table& data)
        : train_input<Task>(data),
          prev_() {}

template <typename Task>
partial_train_input<Task>::partial_train_input() : train_input<Task>(),
                                                   prev_() {}

template <typename Task>
partial_train_input<Task>::partial_train_input(const partial_train_result<Task>& prev,
                                               const table& data)
        : train_input<Task>(data) {
    this->prev_ = prev;
}

template <typename Task>
partial_train_input<Task>::partial_train_input(const partial_train_result<Task>& prev,
                                               const table& data,
                                               const table& responses)
        : train_input<Task>(data, responses) {
    this->prev_ = prev;
}

template <typename Task>
partial_train_input<Task>::partial_train_input(const partial_train_result<Task>& prev,
                                               const partial_train_input<Task>& input)
        : train_input<Task>(input.get_data(), input.get_responses()) {
    this->prev_ = prev;
}

template <typename Task>
partial_train_result<Task>::partial_train_result() : impl_(new partial_train_result_impl<Task>()) {}

template <typename Task>
const table& partial_train_result<Task>::get_partial_xtx() const {
    return impl_->xtx;
}

template <typename Task>
void partial_train_result<Task>::set_partial_xtx_impl(const table& value) {
    impl_->xtx = value;
}

template <typename Task>
const table& partial_train_result<Task>::get_partial_xty() const {
    return impl_->xty;
}

template <typename Task>
void partial_train_result<Task>::set_partial_xty_impl(const table& value) {
    impl_->xty = value;
}

template class ONEDAL_EXPORT train_result<task::regression>;
template class ONEDAL_EXPORT train_input<task::regression>;
template class ONEDAL_EXPORT partial_train_input<task::regression>;
template class ONEDAL_EXPORT partial_train_result<task::regression>;

} // namespace v1
} // namespace oneapi::dal::linear_regression
