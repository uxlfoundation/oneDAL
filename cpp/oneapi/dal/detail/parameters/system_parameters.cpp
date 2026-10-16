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

// --gc-sections drops the reference to std::ios_base_library_init() that GCC 13+
// <iostream> emits to require GLIBCXX_3.4.32. A pointer in a retained section
// keeps that import with every linker; lld ignores -Wl,-u for a shared symbol.
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 13 && !defined(__SYCL_DEVICE_ONLY__) && \
    defined(__has_attribute)
#if __has_attribute(retain)
extern void onedal_ios_base_library_init() __asm__("_ZSt21ios_base_library_initv");
__attribute__((used, retain)) static void (*const onedal_keep_ios_base_library_init)() =
    &onedal_ios_base_library_init;
#endif
#endif

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

std::uint64_t system_parameters::get_l1_cache_size() const {
    return impl_->get_l1_cache_size();
}

std::uint64_t system_parameters::get_l2_cache_size() const {
    return impl_->get_l2_cache_size();
}

std::uint64_t system_parameters::get_ll_cache_size() const {
    return impl_->get_ll_cache_size();
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
