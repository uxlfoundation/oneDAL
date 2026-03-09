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

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::hdbscan {
namespace detail {

result_option_id get_responses_id() {
    return result_option_id::make_by_index(0);
}

result_option_id get_core_observation_indices_id() {
    return result_option_id::make_by_index(1);
}

result_option_id get_core_observations_id() {
    return result_option_id::make_by_index(2);
}

result_option_id get_core_flags_id() {
    return result_option_id::make_by_index(3);
}

template <typename Task>
const result_option_id default_result_options = result_options::responses;

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t min_cluster_size;
    double epsilon;
    bool mem_save_mode;
    result_option_id result_options = default_result_options<Task>;
};

template <typename Task>
class model_impl : public base {
public:
    table centroids;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_cluster_size() const {
    return impl_->min_cluster_size;
}

template <typename Task>
bool descriptor_base<Task>::get_mem_save_mode() const {
    return impl_->mem_save_mode;
}

template <typename Task>
void descriptor_base<Task>::set_min_cluster_size_impl(std::int64_t value) {
    impl_->min_cluster_size = value;
}

template <typename Task>
void descriptor_base<Task>::set_mem_save_mode_impl(bool value) {
    impl_->mem_save_mode = value;
}

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

template class ONEDAL_EXPORT descriptor_base<task::clustering>;

} // namespace v1
} // namespace detail

} // namespace oneapi::dal::hdbscan
