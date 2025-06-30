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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/cpu_info_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info : public cpu_info_iface {
public:
    cpu_info();
    explicit cpu_info(const cpu_extension cpu_extension_);

    cpu_vendor get_cpu_vendor() const override;
    cpu_extension get_top_cpu_extension() const override;

    std::string dump() const override;

private:
    detail::pimpl<cpu_info_iface> impl_;
};

} // namespace v1
using v1::cpu_info;
} // namespace oneapi::dal::detail
