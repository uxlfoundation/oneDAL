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

    std::string dump() const override;

protected:
    std::map<std::string, std::any> info_;

    template <typename T>
    void print(const std::any& value, std::ostringstream& ss) const;

    void print_any(const std::any& value, std::ostringstream& ss) const;
};

} // namespace v1
using v1::cpu_info_impl;
} // namespace oneapi::dal::detail
