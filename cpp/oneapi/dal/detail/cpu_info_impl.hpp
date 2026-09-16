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

#include "oneapi/dal/detail/cpu_info_iface.hpp"

#include <any>
#include <map>
#include <string>

namespace oneapi::dal::detail {
namespace v1 {

std::string ONEDAL_EXPORT to_string(cpu_vendor vendor);
std::string ONEDAL_EXPORT to_string(cpu_extension extension);

class cpu_info_impl : public cpu_info_iface {
public:
    cpu_vendor get_cpu_vendor() const override;

    cpu_extension get_top_cpu_extension() const override;

    cpu_extension get_onedal_cpu_extension() const override;

    uint64_t get_cpu_features() const override;

    std::string dump() const override;

protected:
    std::map<std::string, std::any> info_;
};

} // namespace v1

namespace v2 {

class cpu_info_impl : public cpu_info_iface {
public:
    cpu_vendor get_cpu_vendor() const override;

    cpu_extension get_top_cpu_extension() const override;

    cpu_extension get_onedal_cpu_extension() const override;

    uint64_t get_cpu_features() const override;

    uint64_t get_l1_cache_size() const override;

    uint64_t get_l2_cache_size() const override;

    uint64_t get_l3_cache_size() const override;

    std::string dump() const override;

protected:
    /// Queries the sizes of the CPU data caches and stores them in `info_`.
    /// Called from the constructors of the architecture-specific implementations.
    void detect_cache_sizes();

    std::map<std::string, std::any> info_;
};

} // namespace v2

using v2::cpu_info_impl;
} // namespace oneapi::dal::detail
