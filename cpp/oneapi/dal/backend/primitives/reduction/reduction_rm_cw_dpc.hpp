/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_naive;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_cw_naive {
public:
    using kernel_t = kernel_reduction_rm_cw_naive<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_cw_naive(sycl::queue& q, std::int64_t wg);
    reduction_rm_cw_naive(sycl::queue& q);
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;

private:
    sycl::nd_range<2> get_range(const std::int64_t width) const;
    static kernel_t get_kernel(const Float* input,
                               Float* output,
                               std::int64_t height,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init = true);
    sycl::queue& q_;
    const std::int64_t wg_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_naive_local;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_cw_naive_local {
public:
    using kernel_t = kernel_reduction_rm_cw_naive_local<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_cw_naive_local(sycl::queue& q, std::int64_t wg, std::int64_t lm);
    reduction_rm_cw_naive_local(sycl::queue& q);
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;

private:
    sycl::nd_range<2> get_range(const std::int64_t width) const;
    static kernel_t get_kernel(sycl::handler& h,
                               const Float* input,
                               Float* output,
                               std::int64_t width,
                               std::int64_t height,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init = true);
    sycl::queue& q_;
    const std::int64_t wg_;
    const std::int64_t lm_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_cw_atomic {
public:
    constexpr static inline int max_folding = 8;
    constexpr static inline int block_size = 32;

    reduction_rm_cw_atomic(sycl::queue& q);
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;

private:
    sycl::queue& q_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_blocking;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_cw_blocking {
public:
    using kernel_t = kernel_reduction_rm_cw_blocking<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_cw_blocking(sycl::queue& q);
    reduction_rm_cw_blocking(sycl::queue& q, std::int64_t wg);
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;

private:
    sycl::nd_range<2> get_range(const std::int64_t width) const;
    static kernel_t get_kernel(const Float* input,
                               Float* output,
                               std::int64_t height,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init = true);
    sycl::queue& q_;
    const std::int64_t wg_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_cw {
public:
    using naive_t = reduction_rm_cw_naive<Float, AccT, BinaryOp, UnaryOp>;
    using atomic_t = reduction_rm_cw_atomic<Float, AccT, BinaryOp, UnaryOp>;
    using naive_local_t = reduction_rm_cw_naive_local<Float, AccT, BinaryOp, UnaryOp>;
    using blocking_t = reduction_rm_cw_blocking<Float, AccT, BinaryOp, UnaryOp>;

    reduction_rm_cw(sycl::queue& q);
    enum reduction_method { naive = 0, naive_local = 1, atomic = 2, blocking = 3 };
    reduction_method propose_method(std::int64_t width, std::int64_t height) const;
    sycl::event operator()(reduction_method method,
                           const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           std::int64_t stride,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           std::int64_t stride,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(reduction_method method,
                           const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;
    sycl::event operator()(const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t height,
                           const BinaryOp& binary = BinaryOp{},
                           const UnaryOp& unary = UnaryOp{},
                           const event_vector& deps = {},
                           const bool override_init = true) const;

private:
    sycl::queue& q_;
};

} // namespace oneapi::dal::backend::primitives
