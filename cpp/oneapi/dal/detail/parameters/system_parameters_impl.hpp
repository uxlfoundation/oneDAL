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

#pragma once

#include "oneapi/dal/detail/cpu.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#include <any>
#include <map>

namespace oneapi::dal::detail {
namespace v1 {

class system_parameters_impl {
public:
    explicit system_parameters_impl();

    cpu_extension get_top_enabled_cpu_extension() const;
    std::uint32_t get_max_number_of_threads() const;

#ifdef ONEDAL_DATA_PARALLEL
    std::uint32_t get_max_workgroup_size(sycl::queue& queue) const;
#endif

    std::string dump() const;

#ifdef ONEDAL_DATA_PARALLEL
    std::string dump(sycl::queue& queue) const;
#endif

private:
    std::map<std::string, std::any> sys_info_;

    void print_any(const std::any& value, std::ostringstream& ss) const;
};

} // namespace v1
using v1::system_parameters_impl;
} // namespace oneapi::dal::detail
