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

#include "oneapi/dal/detail/parameters/system_parameters.hpp"

namespace oneapi::dal {

namespace detail {

system_parameters::system_parameters()
        : impl_(detail::pimpl<system_parameters_impl>(std::make_unique<system_parameters_impl>())) {
}

cpu_extension system_parameters::get_top_enabled_cpu_extension() const {
    return impl_->get_top_enabled_cpu_extension();
}

std::uint32_t system_parameters::get_max_number_of_threads() const {
    return impl_->get_max_number_of_threads();
}

std::string system_parameters::dump() const {
    return impl_->dump();
}

#ifdef ONEDAL_DATA_PARALLEL

std::uint32_t system_parameters::get_max_workgroup_size(sycl::queue& queue) const {
    return impl_->get_max_workgroup_size(queue);
}

std::string system_parameters::dump(sycl::queue& queue) const {
    return impl_->dump(queue);
}

#endif

} // namespace detail
} // namespace oneapi::dal
