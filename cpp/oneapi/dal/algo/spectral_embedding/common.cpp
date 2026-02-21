/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/spectral_embedding/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::spectral_embedding::detail {

result_option_id get_embedding_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_eigen_values_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

template <>
result_option_id get_default_result_options<task::compute>() {
    return get_embedding_id();
}

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() {}

    std::int64_t component_count = 0;
    std::int64_t neighbor_count = -1;

    result_option_id result_options = get_default_result_options<Task>();
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_component_count() const {
    return impl_->component_count;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_neighbor_count() const {
    return impl_->neighbor_count;
}

template <typename Task>
void descriptor_base<Task>::set_component_count_impl(std::int64_t component_count) {
    impl_->component_count = component_count;
}

template <typename Task>
void descriptor_base<Task>::set_neighbor_count_impl(std::int64_t neighbor_count) {
    impl_->neighbor_count = neighbor_count;
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

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1

} // namespace oneapi::dal::spectral_embedding::detail
