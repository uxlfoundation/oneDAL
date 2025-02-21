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

#include "oneapi/dal/algo/basic_statistics/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::basic_statistics::detail {

result_option_id get_min_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_max_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

result_option_id get_sum_id() {
    return result_option_id{ result_option_id::make_by_index(2) };
}

result_option_id get_sum_squares_id() {
    return result_option_id{ result_option_id::make_by_index(3) };
}

result_option_id get_sum_squares_centered_id() {
    return result_option_id{ result_option_id::make_by_index(4) };
}

result_option_id get_mean_id() {
    return result_option_id{ result_option_id::make_by_index(5) };
}

result_option_id get_second_order_raw_moment_id() {
    return result_option_id{ result_option_id::make_by_index(6) };
}

result_option_id get_variance_id() {
    return result_option_id{ result_option_id::make_by_index(7) };
}

result_option_id get_standard_deviation_id() {
    return result_option_id{ result_option_id::make_by_index(8) };
}

result_option_id get_variation_id() {
    return result_option_id{ result_option_id::make_by_index(9) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return get_min_id() | get_max_id() | get_sum_id() | get_sum_squares_id() |
           get_sum_squares_centered_id() | get_mean_id() | get_second_order_raw_moment_id() |
           get_variance_id() | get_standard_deviation_id() | get_variation_id();
}

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    result_option_id result_options = get_default_result_options<Task>();
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
result_option_id descriptor_base<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
void descriptor_base<Task>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }

    impl_->result_options = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1
} // namespace oneapi::dal::basic_statistics::detail
