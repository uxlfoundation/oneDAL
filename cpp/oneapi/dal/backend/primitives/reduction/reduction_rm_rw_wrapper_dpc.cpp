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

#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::reduction_rm_rw(sycl::queue& q) : q_{ q } {}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
auto reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::propose_method(std::int64_t width,
                                                                     std::int64_t height) const
    -> reduction_method {
    const std::int64_t max_loop_range = std::numeric_limits<std::int32_t>::max();
    const std::int64_t local_range = width * height;
    if (local_range >= max_loop_range) {
        return reduction_method::blocking;
    }

    const auto device_max_wg = device_max_wg_size(q_);
    if (width < device_max_wg) {
        return reduction_method::narrow;
    }
    else {
        return reduction_method::wide;
    }
    return reduction_method::wide;
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::operator()(
    reduction_method method,
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    // TODO: think about `switch` operator
    if (method == reduction_method::narrow) {
        const narrow_t kernel{ q_ };
        return kernel(input, output, width, height, stride, binary, unary, deps, override_init);
    }
    if (method == reduction_method::wide) {
        const auto device_max_wg = device_max_wg_size(q_);
        const wide_t kernel{ q_, std::min(width, device_max_wg) };
        return kernel(input, output, width, height, stride, binary, unary, deps, override_init);
    }
    if (method == reduction_method::blocking) {
        const blocking_t kernel{ q_ };
        return kernel(input, output, width, height, stride, binary, unary, deps, override_init);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    const auto method = propose_method(width, height);
    return this->
    operator()(method, input, output, width, height, stride, binary, unary, deps, override_init);
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::operator()(
    reduction_method method,
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    return this->
    operator()(method, input, output, width, height, width, binary, unary, deps, override_init);
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw<Float, AccT, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    const auto method = propose_method(width, height);
    return this->
    operator()(method, input, output, width, height, width, binary, unary, deps, override_init);
}

#define INSTANTIATE(F, A, B, U) template class reduction_rm_rw<F, A, B, U>;

#define INSTANTIATE_FLOAT(B, U)                        \
    INSTANTIATE(double, double, B<double>, U<double>); \
    INSTANTIATE(float, float, B<float>, U<float>);

#define INSTANTIATE_BOOL(B, U)                     \
    INSTANTIATE(double, bool, B<bool>, U<double>); \
    INSTANTIATE(float, bool, B<bool>, U<float>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

INSTANTIATE_BOOL(logical_or, isinfornan)
INSTANTIATE_BOOL(logical_or, isinf)

#undef INSTANTIATE_BOOL

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
