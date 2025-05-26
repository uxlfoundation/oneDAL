/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/spmd/exceptions.hpp"

namespace oneapi::dal::preview::spmd {

namespace device_memory_access {
namespace v1 {

struct ONEDAL_EXPORT usm {};
struct ONEDAL_EXPORT none {};
} // namespace v1

using v1::usm;
using v1::none;

} // namespace device_memory_access

namespace v1 {

enum class reduce_op { max, min, sum };

template <typename T>
using enable_if_device_memory_accessible_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm>>;

template <typename T>
using enable_if_device_memory_not_accessible_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::none>>;

} // namespace v1

using v1::reduce_op;
using v1::enable_if_device_memory_accessible_t;
using v1::enable_if_device_memory_not_accessible_t;

} // namespace oneapi::dal::preview::spmd
