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
class kernel_reduction_rm_rw_wide;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw_wide {
public:
    using kernel_t = kernel_reduction_rm_rw_wide<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_rw_wide(sycl::queue& q, std::int64_t wg);
    reduction_rm_rw_wide(sycl::queue& q);
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
    sycl::nd_range<2> get_range(std::int64_t height) const;
    static kernel_t get_kernel(const Float* input,
                               Float* output,
                               std::int64_t width,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init);
    sycl::queue& q_;
    const std::int64_t wg_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_rw_blocking;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw_blocking {
public:
    using kernel_t = kernel_reduction_rm_rw_blocking<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_rw_blocking(sycl::queue& q, std::int64_t wg);
    reduction_rm_rw_blocking(sycl::queue& q);
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
    sycl::nd_range<2> get_range(std::int64_t height) const;
    static kernel_t get_kernel(const Float* input,
                               Float* output,
                               std::int64_t width,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init);
    sycl::queue& q_;
    const std::int64_t wg_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_rw_narrow;

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw_narrow {
public:
    using kernel_t = kernel_reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_rw_narrow(sycl::queue& q, std::int64_t wg);
    reduction_rm_rw_narrow(sycl::queue& q);
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
    sycl::nd_range<1> get_range(std::int64_t height) const;
    static kernel_t get_kernel(const Float* input,
                               Float* output,
                               std::int64_t width,
                               std::int64_t height,
                               std::int64_t stride,
                               const BinaryOp& binary,
                               const UnaryOp& unary,
                               const bool override_init);
    sycl::queue& q_;
    const std::int64_t wg_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
class reduction_rm_rw {
public:
    using narrow_t = reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>;
    using wide_t = reduction_rm_rw_wide<Float, AccT, BinaryOp, UnaryOp>;
    using blocking_t = reduction_rm_rw_blocking<Float, AccT, BinaryOp, UnaryOp>;
    reduction_rm_rw(sycl::queue& q);
    enum reduction_method { wide, narrow, blocking };
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
