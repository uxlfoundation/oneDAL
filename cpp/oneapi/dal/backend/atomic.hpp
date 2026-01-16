/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::backend {

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_global_add(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::global_space> atomic_var(
        *ptr);
    return atomic_var.fetch_add(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_global_sum(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::global_space> atomic_var(
        *ptr);
    auto old = atomic_var.fetch_add(operand);
    return old + operand;
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_global_min(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::global_space> atomic_var(
        *ptr);
    return atomic_var.fetch_min(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_global_max(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::global_space> atomic_var(
        *ptr);
    return atomic_var.fetch_max(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_global_cmpxchg(T* ptr, T expected, T desired) {
    T expected_ = expected;
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::global_space> atomic_var(
        *ptr);
    atomic_var.compare_exchange_weak(expected_, desired, mem_order, mem_scope);
    return expected_;
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_local_add(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::local_space> atomic_var(
        *ptr);
    return atomic_var.fetch_add(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_local_sum(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::local_space> atomic_var(
        *ptr);
    auto old = atomic_var.fetch_add(operand);
    return old + operand;
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_local_min(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::local_space> atomic_var(
        *ptr);
    return atomic_var.fetch_min(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_local_max(T* ptr, T operand) {
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::local_space> atomic_var(
        *ptr);
    return atomic_var.fetch_max(operand);
}

template <typename T,
          sycl::memory_order mem_order = sycl::memory_order::relaxed,
          sycl::memory_scope mem_scope = sycl::memory_scope::device>
inline T atomic_local_cmpxchg(T* ptr, T expected, T desired) {
    T expected_temp = expected;
    sycl::atomic_ref<T, mem_order, mem_scope, sycl::access::address_space::local_space> atomic_var(
        *ptr);
    atomic_var.compare_exchange_weak(expected_temp, desired, mem_order, mem_scope);
    return expected_temp;
}

} // namespace oneapi::dal::backend

#endif
