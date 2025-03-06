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

#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include <iostream>
namespace oneapi::dal::covariance {

namespace detail::v1 {

template <typename Task>
class compute_input_impl : public base {
public:
    compute_input_impl() : data(table()){};
    compute_input_impl(const table& data) : data(data){};
    table data;
};

template <typename Task>
class compute_result_impl : public base {
public:
    table cov_matrix;
    table cor_matrix;
    table means;
    result_option_id options;
};

template <typename Task>
class partial_compute_result_impl : public base {
public:
    table nobs;
    table crossproduct;
    table sums;
};

/// Structure that contains all the hyperparameters of the covariance algorithm.
///
/// @tparam Task    The variant of the computations.
///                 Covariance algorithm supports only :expr:`compute`.
template <typename Task>
struct compute_parameters_impl : public base {
    /// To compute the variance-covariance matrix input data set is being split into blocks of rows.
    /// This value defines the default number of rows in the block on CPU.
    std::int64_t cpu_macro_block = 140l;
};

template <typename Task>
compute_parameters<Task>::compute_parameters() : impl_(new compute_parameters_impl<Task>{}) {}

/// Choose the number of rows in the data block used in variance-covariance matrix computations on CPU.
///
/// @tparam Task    The variant of the computations.
///                 Covariance algorithm supports only :expr:`compute`.
template <typename Task>
std::int64_t compute_parameters<Task>::get_cpu_macro_block() const {
    return impl_->cpu_macro_block;
}

template <typename Task>
void compute_parameters<Task>::set_cpu_macro_block_impl(std::int64_t val) {
    std::cout << "macro block value=" << val << std::endl;
    impl_->cpu_macro_block = val;
}

template class ONEDAL_EXPORT compute_parameters<task::compute>;

} // namespace detail::v1

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;
using detail::v1::partial_compute_result_impl;
using detail::v1::compute_parameters;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input() : impl_(new compute_input_impl<Task>{}) {}

template <typename Task>
compute_input<Task>::compute_input(const table& data) : impl_(new compute_input_impl<Task>(data)) {}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_cov_matrix() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::cov_matrix)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->cov_matrix;
}

template <typename Task>
void compute_result<Task>::set_cov_matrix_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::cov_matrix)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->cov_matrix = value;
}
template <typename Task>
const table& compute_result<Task>::get_cor_matrix() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::cor_matrix)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->cor_matrix;
}

template <typename Task>
void compute_result<Task>::set_cor_matrix_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::cor_matrix)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->cor_matrix = value;
}

template <typename Task>
const table& compute_result<Task>::get_means() const {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::means)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    return impl_->means;
}

template <typename Task>
void compute_result<Task>::set_means_impl(const table& value) {
    using msg = dal::detail::error_messages;
    if (!get_result_options().test(result_options::means)) {
        throw domain_error(msg::this_result_is_not_enabled_via_result_options());
    }
    impl_->means = value;
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
const result_option_id& compute_result<Task>::get_result_options() const {
    return impl_->options;
}

template <typename Task>
void compute_result<Task>::set_result_options_impl(const result_option_id& value) {
    impl_->options = value;
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
const table& partial_compute_result<Task>::get_partial_crossproduct() const {
    return impl_->crossproduct;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_crossproduct_impl(const table& value) {
    impl_->crossproduct = value;
}
template <typename Task>
const table& partial_compute_result<Task>::get_partial_sum() const {
    return impl_->sums;
}

template <typename Task>
void partial_compute_result<Task>::set_partial_sum_impl(const table& value) {
    impl_->sums = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;
template class ONEDAL_EXPORT compute_result<task::compute>;
template class ONEDAL_EXPORT partial_compute_input<task::compute>;
template class ONEDAL_EXPORT partial_compute_result<task::compute>;

} // namespace v1
} // namespace oneapi::dal::covariance
