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

#include <variant>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class array_via_policy {
public:
    array_via_policy() = delete;

    template <typename... Args>
    static array<T> wrap(const default_host_policy& policy, Args&&... args) {
        return array<T>{ std::forward<Args>(args)... };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    static array<T> wrap(const data_parallel_policy& policy, Args&&... args) {
        return array<T>{ policy.get_queue(), std::forward<Args>(args)... };
    }
#endif
};

template <typename T, typename Body>
inline auto dispatch_by_policy(const dal::array<T>& data, Body&& body) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto optional_queue = data.get_queue();
    if (optional_queue) {
        return body(data_parallel_policy{ optional_queue.value() });
    }
#endif
    return body(default_host_policy{});
}

template <typename T, typename U>
inline dal::array<T> reinterpret_array_cast(const dal::array<U>& ary) {
    if (ary.get_size() % sizeof(T) > 0) {
        throw invalid_argument{ error_messages::incompatible_array_reinterpret_cast_types() };
    }

    const std::int64_t new_count = ary.get_size() / sizeof(T);
    if (ary.has_mutable_data()) {
        T* mutable_ptr = reinterpret_cast<T*>(ary.get_mutable_data());
        return dal::array<T>{ ary, mutable_ptr, new_count };
    }
    else {
        const T* immutable_ptr = reinterpret_cast<const T*>(ary.get_data());
        return dal::array<T>{ ary, immutable_ptr, new_count };
    }
}

template <typename T>
inline dal::array<T> discard_mutable_data(const dal::array<T>& ary) {
    if (!ary.has_mutable_data()) {
        return ary;
    }
    return dal::array<T>{ ary, ary.get_data(), ary.get_count() };
}

} // namespace v1
using v1::array_via_policy;
using v1::dispatch_by_policy;
using v1::reinterpret_array_cast;
using v1::discard_mutable_data;

namespace v2 {

template <typename T>
inline void copy_impl(detail::array_impl<T>& dst, const detail::array_impl<T>& src) {
    const auto size_in_bytes = src.get_size_in_bytes();
    ONEDAL_ASSERT(size_in_bytes <= dst.get_size_in_bytes());

    auto* const dst_ptr = reinterpret_cast<void*>(dst.get_mutable_data());
    const auto* const src_ptr = reinterpret_cast<const void*>(src.get_data());

    const auto copy_visitor = [&](const auto& dst_policy, const auto& src_policy) {
        memcpy(dst_policy, src_policy, dst_ptr, src_ptr, size_in_bytes);
    };

    std::visit(copy_visitor, dst.get_policy(), src.get_policy());
}

template <typename Policy, typename T, typename Alloc>
inline detail::array_impl<T> copy_impl(const Policy& policy,
                                       const detail::array_impl<T>& src,
                                       const Alloc& alloc) {
    using res_t = detail::array_impl<T>;
    auto res = res_t::empty_unique(policy, src.get_count(), alloc);

    copy_impl(*res, src);

    return res_t{ std::move(*res) };
}

template <typename Policy, typename T>
inline detail::array_impl<T> copy_impl(const Policy& policy, const detail::array_impl<T>& src) {
    const auto alloc = make_policy_allocator<Policy, T>(policy);
    return copy_impl(policy, src, alloc);
}

template <typename T>
inline void copy(dal::array<T>& dst, const dal::array<T>& src) {
    constexpr detail::pimpl_accessor accessor;
    auto& dst_pimpl = accessor.get_pimpl(dst);
    const auto& src_pimpl = accessor.get_pimpl(src);

    copy_impl(*dst_pimpl, *src_pimpl);
}

template <typename Policy, typename T>
inline auto copy(const Policy& policy, const dal::array<T>& src) {
    constexpr detail::pimpl_accessor accessor;
    const auto& pimpl = accessor.get_pimpl(src);
    array_impl<T> impl = copy_impl(policy, *pimpl);
    return array<T>{ new array_impl<T>{ std::move(impl) } };
}

} // namespace v2

using v2::copy;
using v2::copy_impl;

} // namespace oneapi::dal::detail
