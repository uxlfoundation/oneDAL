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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/cpu_info.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class global_context_iface : public base {
public:
    virtual const cpu_info_iface& get_cpu_info() const = 0;
};

class global_context : public global_context_iface {
public:
    ONEDAL_EXPORT static const global_context_iface& get_global_context();

    virtual ~global_context() = default;

    global_context(const global_context& ctx) = delete;
    global_context& operator=(const global_context& ctx) = delete;

private:
    global_context() {}
};

} // namespace v1

namespace v2 {

/// Narrows `get_cpu_info()` down to `v2::cpu_info_iface`, which additionally exposes
/// the sizes of the CPU caches. As with `cpu_info_iface`, the interface is derived from
/// the v1 one instead of being modified in place, so that the layout of the
/// `v1::global_context_iface` virtual table, and thus the library ABI, stays unchanged.
class global_context_iface : public v1::global_context_iface {
public:
    const v2::cpu_info_iface& get_cpu_info() const override = 0;
};

class global_context : public global_context_iface {
public:
    ONEDAL_EXPORT static const global_context_iface& get_global_context();

    virtual ~global_context() = default;

    global_context(const global_context& ctx) = delete;
    global_context& operator=(const global_context& ctx) = delete;

private:
    global_context() {}
};

} // namespace v2

using v2::global_context;
using v2::global_context_iface;
} // namespace oneapi::dal::detail
