/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <type_traits>
#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/cpu.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class host_policy_impl;
class data_parallel_policy_impl;

template <typename T>
struct is_execution_policy : std::bool_constant<false> {};

template <typename T>
struct is_local_policy : std::bool_constant<false> {};

template <typename T>
struct is_distributed_policy : std::bool_constant<false> {};

template <typename T>
struct is_host_policy : std::bool_constant<false> {};

template <typename T>
struct is_data_parallel_policy : std::bool_constant<false> {};

template <typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

template <typename T>
inline constexpr bool is_local_policy_v = is_local_policy<T>::value;

template <typename T>
inline constexpr bool is_distributed_policy_v = is_distributed_policy<T>::value;

template <typename T>
inline constexpr bool is_host_policy_v = is_host_policy<T>::value;

template <typename T>
inline constexpr bool is_data_parallel_policy_v = is_data_parallel_policy<T>::value;

class ONEDAL_EXPORT default_host_policy {};

class ONEDAL_EXPORT host_policy : public base {
public:
    host_policy();

    static const host_policy& get_default() {
        const static host_policy instance;
        return instance;
    }

    cpu_extension get_enabled_cpu_extensions() const noexcept;

    auto& set_enabled_cpu_extensions(const cpu_extension& extensions) {
        set_enabled_cpu_extensions_impl(extensions);
        return *this;
    }

private:
    void set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept;

    pimpl<host_policy_impl> impl_;
};

template <>
struct is_execution_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_local_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_host_policy<host_policy> : std::bool_constant<true> {};

#ifdef ONEDAL_DATA_PARALLEL
class ONEDAL_EXPORT data_parallel_policy : public base {
public:
    data_parallel_policy(const sycl::queue& queue) : queue_(queue) {
        init_impl(queue);
    }

    sycl::queue& get_queue() const noexcept {
        return queue_;
    }

private:
    void init_impl(const sycl::queue& queue);

private:
    mutable sycl::queue queue_;
    pimpl<data_parallel_policy_impl> impl_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_execution_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_local_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_data_parallel_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

} // namespace v1

using v1::is_execution_policy_v;
using v1::is_local_policy_v;
using v1::is_distributed_policy_v;
using v1::is_host_policy_v;
using v1::is_data_parallel_policy_v;

using v1::default_host_policy;
using v1::host_policy;

#ifdef ONEDAL_DATA_PARALLEL
using v1::data_parallel_policy;
#endif

} // namespace oneapi::dal::detail
