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

#include "oneapi/dal/algo/basic_statistics/compute_types.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::basic_statistics {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl() : data(table()){};
    compute_input_impl(const table& data) : data(data) {}
    compute_input_impl(const table& data, const table& weights) : data(data), weights(weights) {}
    table data, weights;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    table min;
    table max;
    table sum;
    table sum2;
    table sum2cent;
    table mean;
    table sorm;
    table varc;
    table stdev;
    table vart;

    result_option_id options;
};

template <typename Task>
class detail::v1::partial_compute_result_impl : public base {
public:
    table nobs;
    table partial_min;
    table partial_max;
    table partial_sum;
    table partial_sum_squares;
    table partial_sum_squares_centered;
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;
using detail::v1::partial_compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input() : impl_(new compute_input_impl<Task>{}) {}

template <typename Task>
compute_input<Task>::compute_input(const table& data) : impl_(new compute_input_impl<Task>(data)) {}

template <typename Task>
compute_input<Task>::compute_input(const table& data, const table& weights)
        : impl_(new compute_input_impl<Task>(data, weights)) {}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& compute_input<Task>::get_weights() const {
    return impl_->weights;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void compute_input<Task>::set_weights_impl(const table& value) {
    impl_->weights = value;
}

using msg = dal::detail::error_messages;

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_min() const {
    if (!get_result_options().test(result_options::min)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->min;
}

template <typename Task>
const table& compute_result<Task>::get_max() const {
    if (!get_result_options().test(result_options::max)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->max;
}

template <typename Task>
const table& compute_result<Task>::get_sum() const {
    if (!get_result_options().test(result_options::sum)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->sum;
}

template <typename Task>
const table& compute_result<Task>::get_sum_squares() const {
    if (!get_result_options().test(result_options::sum_squares)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->sum2;
}

template <typename Task>
const table& compute_result<Task>::get_sum_squares_centered() const {
    if (!get_result_options().test(result_options::sum_squares_centered)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->sum2cent;
}

template <typename Task>
const table& compute_result<Task>::get_mean() const {
    if (!get_result_options().test(result_options::mean)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->mean;
}

template <typename Task>
const table& compute_result<Task>::get_second_order_raw_moment() const {
    if (!get_result_options().test(result_options::second_order_raw_moment)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->sorm;
}

template <typename Task>
const table& compute_result<Task>::get_variance() const {
    if (!get_result_options().test(result_options::variance)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->varc;
}

template <typename Task>
const table& compute_result<Task>::get_standard_deviation() const {
    if (!get_result_options().test(result_options::standard_deviation)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->stdev;
}

template <typename Task>
const table& compute_result<Task>::get_variation() const {
    if (!get_result_options().test(result_options::variation)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->vart;
}

template <typename Task>
const result_option_id& compute_result<Task>::get_result_options() const {
    return impl_->options;
}

template <typename Task>
void compute_result<Task>::set_min_impl(const table& value) {
    if (!get_result_options().test(result_options::min)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->min = value;
}

template <typename Task>
void compute_result<Task>::set_max_impl(const table& value) {
    if (!get_result_options().test(result_options::max)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->max = value;
}

template <typename Task>
void compute_result<Task>::set_sum_impl(const table& value) {
    if (!get_result_options().test(result_options::sum)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->sum = value;
}

template <typename Task>
void compute_result<Task>::set_sum_squares_impl(const table& value) {
    if (!get_result_options().test(result_options::sum_squares)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->sum2 = value;
}

template <typename Task>
void compute_result<Task>::set_sum_squares_centered_impl(const table& value) {
    if (!get_result_options().test(result_options::sum_squares_centered)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->sum2cent = value;
}

template <typename Task>
void compute_result<Task>::set_mean_impl(const table& value) {
    if (!get_result_options().test(result_options::mean)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->mean = value;
}

template <typename Task>
void compute_result<Task>::set_second_order_raw_moment_impl(const table& value) {
    if (!get_result_options().test(result_options::second_order_raw_moment)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->sorm = value;
}

template <typename Task>
void compute_result<Task>::set_variance_impl(const table& value) {
    if (!get_result_options().test(result_options::variance)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->varc = value;
}

template <typename Task>
void compute_result<Task>::set_standard_deviation_impl(const table& value) {
    if (!get_result_options().test(result_options::standard_deviation)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->stdev = value;
}

template <typename Task>
void compute_result<Task>::set_variation_impl(const table& value) {
    if (!get_result_options().test(result_options::variation)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->vart = value;
}

template <typename Task>
void compute_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->options = value;
}

template <typename Task>
partial_compute_input<Task>::partial_compute_input(const table& data)
        : compute_input<Task>(data),
          prev_() {}

template <typename Task>
partial_compute_input<Task>::partial_compute_input() : compute_input<Task>(),
                                                       prev_() {}

template <typename Task>
partial_compute_input<Task>::partial_compute_input(const partial_compute_result<Task>& prev,
                                                   const table& data)
        : compute_input<Task>(data) {
    this->prev_ = prev;
}

template <typename Task>
partial_compute_input<Task>::partial_compute_input(const partial_compute_result<Task>& prev,
                                                   const table& data,
                                                   const table& weights)
        : compute_input<Task>(data, weights) {
    this->prev_ = prev;
}

template <typename Task>
const table& partial_compute_result<Task>::get_partial_n_rows() const {
    return impl_->nobs;
}

template <typename Task>
partial_compute_result<Task>::partial_compute_result()
        : impl_(new partial_compute_result_impl<Task>()) {}

template <typename Task>
void partial_compute_result<Task>::set_partial_n_rows_impl(const table& value) {
    impl_->nobs = value;
}

template <typename Task>
const table& partial_compute_result<Task>::get_partial_min() const {
    return impl_->partial_min;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_min_impl(const table& value) {
    impl_->partial_min = value;
}
template <typename Task>
const table& partial_compute_result<Task>::get_partial_max() const {
    return impl_->partial_max;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_max_impl(const table& value) {
    impl_->partial_max = value;
}

template <typename Task>
const table& partial_compute_result<Task>::get_partial_sum() const {
    return impl_->partial_sum;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_sum_impl(const table& value) {
    impl_->partial_sum = value;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_sum_squares_impl(const table& value) {
    impl_->partial_sum_squares = value;
}
template <typename Task>
const table& partial_compute_result<Task>::get_partial_sum_squares() const {
    return impl_->partial_sum_squares;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_sum_squares_centered_impl(const table& value) {
    impl_->partial_sum_squares_centered = value;
}

template <typename Task>
const table& partial_compute_result<Task>::get_partial_sum_squares_centered() const {
    return impl_->partial_sum_squares_centered;
}
template class ONEDAL_EXPORT compute_input<task::compute>;
template class ONEDAL_EXPORT compute_result<task::compute>;
template class ONEDAL_EXPORT partial_compute_result<task::compute>;
template class ONEDAL_EXPORT partial_compute_input<task::compute>;

} // namespace v1
} // namespace oneapi::dal::basic_statistics
