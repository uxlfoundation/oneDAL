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

#include "oneapi/dal/detail/cpu_info_impl.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

#include <sstream>

namespace oneapi::dal::detail {
namespace v1 {

std::string ONEDAL_EXPORT to_string(cpu_vendor vendor) {
    std::string vendor_str;
    switch (vendor) {
        case cpu_vendor::unknown: vendor_str = std::string("Unknown"); break;
        case cpu_vendor::intel: vendor_str = std::string("Intel"); break;
        case cpu_vendor::amd: vendor_str = std::string("AMD"); break;
        case cpu_vendor::arm: vendor_str = std::string("Arm"); break;
        case cpu_vendor::riscv64: vendor_str = std::string("RISCV-V"); break;
    }
    return vendor_str;
}

std::string ONEDAL_EXPORT to_string(cpu_extension extension) {
    std::string extension_str;
    switch (extension) {
        case cpu_extension::none: extension_str = std::string("none"); break;
#if defined(TARGET_X86_64)
        case cpu_extension::sse2: extension_str = std::string("sse2"); break;
        case cpu_extension::sse42: extension_str = std::string("sse42"); break;
        case cpu_extension::avx2: extension_str = std::string("avx2"); break;
        case cpu_extension::avx512: extension_str = std::string("avx512"); break;
#elif defined(TARGET_ARM)
        case cpu_extension::sve: extension_str = std::string("sve"); break;
#elif defined(TARGET_RISCV64)
        case cpu_extension::rv64: extension_str = std::string("rv64"); break;
#endif
    }
    return extension_str;
}

template <typename T>
void to_stream(const std::any& value, std::ostream& ss) {
    T typed_value = std::any_cast<T>(value);
    ss << to_string(typed_value);
}

void any_to_stream(const std::any& value, std::ostream& ss) {
    const std::type_info& ti = value.type();
    if (ti == typeid(cpu_extension)) {
        to_stream<cpu_extension>(value, ss);
    }
    else if (ti == typeid(cpu_vendor)) {
        to_stream<cpu_vendor>(value, ss);
    }
    else {
        throw unimplemented{ dal::detail::error_messages::unsupported_data_type() };
    }
}

void cpu_features_to_stream(const std::any& value, std::ostream& ss) {
    std::uint64_t cpu_features = std::any_cast<std::uint64_t>(value);
    if (cpu_features == 0) {
        const auto entry = cpu_feature_map.find(0);
        if (entry == cpu_feature_map.end()) {
            throw invalid_argument{ error_messages::invalid_key() };
        }
        ss << entry->second;
    }
    else {
        for (const auto& [key, feature] : cpu_feature_map) {
            if (key && (cpu_features & key) != 0) {
                ss << feature << ", ";
            }
        }
    }
}

cpu_vendor cpu_info_impl::get_cpu_vendor() const {
    const auto entry = info_.find("vendor");
    if (entry == info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<detail::cpu_vendor>(entry->second);
}

cpu_extension cpu_info_impl::get_top_cpu_extension() const {
    const auto entry = info_.find("top_cpu_extension");
    if (entry == info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<cpu_extension>(entry->second);
}

cpu_extension cpu_info_impl::get_onedal_cpu_extension() const {
    const auto entry = info_.find("onedal_cpu_extension");
    if (entry == info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<cpu_extension>(entry->second);
}

uint64_t cpu_info_impl::get_cpu_features() const {
    const auto entry = info_.find("cpu_features");
    if (entry == info_.end()) {
        throw invalid_argument{ error_messages::invalid_key() };
    }
    return std::any_cast<uint64_t>(entry->second);
}

std::string cpu_info_impl::dump() const {
    std::ostringstream ss;
    for (auto const& [name, value] : info_) {
        ss << name << " : ";
        if (name == "cpu_features") {
            cpu_features_to_stream(value, ss);
        }
        else {
            any_to_stream(value, ss);
        }
        ss << "; ";
    }
    return std::move(ss).str();
}

} // namespace v1
} // namespace oneapi::dal::detail
