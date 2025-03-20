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

/// @file Data access interfaces needed for binary backward compatibility with
/// the oneAPI Data Analytics Library (oneDAL) 2021.1. This file should be removed in 2022.1.

#pragma once

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::detail {
namespace v1 {

struct row_block {
    range rows;

    row_block(const range& rows) : rows(rows) {}
};

struct column_values_block {
    std::int64_t column_index;
    range rows;

    column_values_block(std::int64_t idx, const range& rows) : column_index(idx), rows(rows) {}
};

template <typename Policy, template <typename> typename Allocator>
struct access_iface {
    using array_f32 = dal::array<float>;
    using array_f64 = dal::array<double>;
    using array_i32 = dal::array<std::int32_t>;

    using alloc_f32 = Allocator<float>;
    using alloc_f64 = Allocator<double>;
    using alloc_i32 = Allocator<std::int32_t>;

    virtual ~access_iface() {}

    virtual void pull(const Policy&, array_f32&, const row_block&, const alloc_f32&) const = 0;
    virtual void pull(const Policy&, array_f64&, const row_block&, const alloc_f64&) const = 0;
    virtual void pull(const Policy&, array_i32&, const row_block&, const alloc_i32&) const = 0;
    virtual void pull(const Policy&,
                      array_f32&,
                      const column_values_block&,
                      const alloc_f32&) const = 0;
    virtual void pull(const Policy&,
                      array_f64&,
                      const column_values_block&,
                      const alloc_f64&) const = 0;
    virtual void pull(const Policy&,
                      array_i32&,
                      const column_values_block&,
                      const alloc_i32&) const = 0;

    virtual void push(const Policy&, const array_f32&, const row_block&) = 0;
    virtual void push(const Policy&, const array_f64&, const row_block&) = 0;
    virtual void push(const Policy&, const array_i32&, const row_block&) = 0;
    virtual void push(const Policy&, const array_f32&, const column_values_block&) = 0;
    virtual void push(const Policy&, const array_f64&, const column_values_block&) = 0;
    virtual void push(const Policy&, const array_i32&, const column_values_block&) = 0;
};

using access_iface_host = access_iface<default_host_policy, host_allocator>;

#ifdef ONEDAL_DATA_PARALLEL
using access_iface_dpc = access_iface<data_parallel_policy, data_parallel_allocator>;
#endif

class access_provider_iface {
public:
    virtual ~access_provider_iface() {}
};

} // namespace v1

using v1::row_block;
using v1::column_values_block;
using v1::access_iface;
using v1::access_iface_host;
using v1::access_provider_iface;

#ifdef ONEDAL_DATA_PARALLEL
using v1::access_iface_dpc;
#endif

} // namespace oneapi::dal::detail
