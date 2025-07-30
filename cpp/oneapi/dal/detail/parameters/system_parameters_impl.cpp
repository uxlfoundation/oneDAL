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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/cpu_info_impl.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/parameters/system_parameters_impl.hpp"
#include <daal/src/services/service_defines.h>
#include <daal/include/services/internal/daal_kernel_defines.h>

#include <any>
#include <sstream>

namespace oneapi::dal::detail {
namespace v1 {

system_parameters_impl::system_parameters_impl() {
    using daal::services::Environment;
    Environment* env = Environment::getInstance();
    if (!env) {
        throw std::runtime_error("Environment not initialized, cannot get processor info.");
    }
    // Call to `getCpuId` changes global settings, in particular,
    // changes default number of threads in the threading layer
    const int cpuid = env->getCpuId();

    sys_info_.insert(
        { "top_enabled_cpu_extension", std::make_any<cpu_extension>(from_daal_cpu_type(cpuid)) });
    sys_info_.insert(
        { "max_number_of_threads", std::make_any<std::uint32_t>(env->getNumberOfThreads()) });
}

cpu_extension system_parameters_impl::get_top_enabled_cpu_extension() const {
    const auto entry = sys_info_.find("top_enabled_cpu_extension");
    if (entry == sys_info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<cpu_extension>(entry->second);
}

std::uint32_t system_parameters_impl::get_max_number_of_threads() const {
    const auto entry = sys_info_.find("max_number_of_threads");
    if (entry == sys_info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<std::uint32_t>(entry->second);
}

void system_parameters_impl::print_any(const std::any& value, std::ostringstream& ss) const {
    const std::type_info& ti = value.type();
    if (ti == typeid(cpu_extension)) {
        ss << to_string(std::any_cast<cpu_extension>(value));
    }
    else if (ti == typeid(std::uint32_t)) {
        ss << std::any_cast<std::uint32_t>(value);
    }
    else {
        throw unimplemented{ dal::detail::error_messages::unsupported_data_type() };
    }
}

std::string system_parameters_impl::dump() const {
    std::ostringstream ss;
    for (auto const& [name, value] : sys_info_) {
        ss << name << " : ";
        print_any(value, ss);
        ss << "; ";
    }
    return std::move(ss).str();
}

#ifdef ONEDAL_DATA_PARALLEL

std::uint32_t system_parameters_impl::get_max_workgroup_size(sycl::queue& queue) const {
    return dal::backend::device_max_wg_size(queue);
}

std::string system_parameters_impl::dump(sycl::queue& queue) const {
    std::ostringstream ss;
    ss << "max_workgroup_size"
       << " : " << get_max_workgroup_size(queue) << "; ";
    ss << dump();
    return std::move(ss).str();
}

#endif

} // namespace v1
} // namespace oneapi::dal::detail
