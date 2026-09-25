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

#include "oneapi/dal/detail/global_context.hpp"

namespace oneapi::dal::detail {
namespace v1 {

/// The class stays in `v1` so that the exported constructor keeps its mangled name, but
/// it implements `v2::global_context_iface`. Since the v2 interfaces only derive from the
/// v1 ones, this neither adds virtual table slots nor changes the object layout.
class global_context_impl : public v2::global_context_iface {
public:
    global_context_impl();

    const v2::cpu_info_iface &get_cpu_info() const override {
        return cpu_info_;
    }

private:
    v2::cpu_info cpu_info_;
};

} // namespace v1
using v1::global_context_impl;
} // namespace oneapi::dal::detail
