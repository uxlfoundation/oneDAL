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

#include <cmath>
#include <type_traits>

namespace oneapi::dal::backend::primitives {

struct reduce_unary_op_tag;

template <typename T>
struct identity {
    using tag_t = reduce_unary_op_tag;
    T operator()(const T& arg) const {
        return arg;
    }
};

template <typename T>
struct abs {
    using tag_t = reduce_unary_op_tag;
    T operator()(const T& arg) const {
#ifdef __SYCL_DEVICE_ONLY__
        return sycl::fabs(arg);
#else
        return std::abs(arg);
#endif
    }
};

template <typename T>
struct square {
    using tag_t = reduce_unary_op_tag;
    T operator()(const T& arg) const {
        return (arg * arg);
    }
};

template <typename T>
struct isinfornan {
    using tag_t = reduce_unary_op_tag;
    bool operator()(const T& arg) const {
#ifdef ONEDAL_DATA_PARALLEL
        return static_cast<T>(sycl::isinf(arg) || sycl::isnan(arg));
#else
        return static_cast<T>(isinf(arg) || (arg != arg));
#endif
    }
};

template <typename T>
struct isinf {
    using tag_t = reduce_unary_op_tag;
    bool operator()(const T& arg) const {
#ifdef ONEDAL_DATA_PARALLEL
        return static_cast<T>(sycl::isinf(arg));
#else
        return static_cast<T>(isinf(arg));
#endif
    }
};

struct reduce_binary_op_tag;

template <typename T>
struct sum {
    using tag_t = reduce_binary_op_tag;
    constexpr static inline T init_value = 0;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ext::oneapi::plus<T> native{};
#else
    constexpr static inline std::plus<T> native{};
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename T>
struct max {
    using tag_t = reduce_binary_op_tag;
    constexpr static inline T init_value = std::numeric_limits<T>::lowest();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ext::oneapi::maximum<T> native{};
#else
    constexpr static inline auto native = [](const T& a, const T& b) {
        return std::max(a, b);
    };
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename T>
struct min {
    using tag_t = reduce_binary_op_tag;
    constexpr static inline T init_value = std::numeric_limits<T>::max();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ext::oneapi::minimum<T> native{};
#else
    constexpr static inline auto native = [](const T& a, const T& b) {
        return std::min(a, b);
    };
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename T>
struct logical_or {
    using tag_t = reduce_binary_op_tag;
    constexpr static inline T init_value = false;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::logical_or<T> native{};
#else
    constexpr static inline std::logical_or<T> native{};
};
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename Float, typename BinaryOp>
constexpr bool is_typed_sum_op_v = std::is_same_v<sum<Float>, BinaryOp>;

template <typename Float, typename BinaryOp>
constexpr bool is_typed_min_op_v = std::is_same_v<min<Float>, BinaryOp>;

template <typename Float, typename BinaryOp>
constexpr bool is_typed_max_op_v = std::is_same_v<max<Float>, BinaryOp>;

template <typename BinaryOp>
using bin_op_t = std::remove_const_t<decltype(BinaryOp::init_value)>;

template <typename BinaryOp>
constexpr bool is_sum_op_v = is_typed_sum_op_v<bin_op_t<BinaryOp>, BinaryOp>;

template <typename BinaryOp>
constexpr bool is_min_op_v = is_typed_min_op_v<bin_op_t<BinaryOp>, BinaryOp>;

template <typename BinaryOp>
constexpr bool is_max_op_v = is_typed_max_op_v<bin_op_t<BinaryOp>, BinaryOp>;

#ifdef ONEDAL_DATA_PARALLEL

template <typename BinaryOp, typename T = bin_op_t<BinaryOp>>
inline T atomic_binary_op(T* ptr, T val) {
    sycl::atomic_ref<T,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        atomic_ref(*ptr);
    if constexpr (is_sum_op_v<BinaryOp>) {
        return atomic_ref.fetch_add(val);
    }
    else if constexpr (is_min_op_v<BinaryOp>) {
        return atomic_ref.fetch_min(val);
    }
    else if constexpr (is_max_op_v<BinaryOp>) {
        return atomic_ref.fetch_max(val);
    }
    else {
        return val;
    }
}

#endif

} // namespace oneapi::dal::backend::primitives
