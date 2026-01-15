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
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename BinaryOp, typename UnaryOp>
class kernel_reduction_rm_cw_naive {
public:
    kernel_reduction_rm_cw_naive(const Float* input,
                                 Float* output,
                                 std::int64_t height,
                                 std::int32_t lstride,
                                 const BinaryOp& binary,
                                 const UnaryOp& unary,
                                 const bool override_init)
            : input_{ input },
              output_{ output },
              unary_{ unary },
              binary_{ binary },
              height_{ height },
              lstride_{ lstride },
              override_init_{ override_init } {}

    void operator()(sycl::nd_item<2> it) const {
        // Common for whole WG
        const auto col_idx = it.get_global_id(0);
        const auto loc_idx = it.get_global_id(1);
        const auto range = it.get_global_range(1);
        // Exclusive for EU
        Float acc = (override_init_ || (loc_idx != 0)) ? //
                        binary_.init_value
                                                       : output_[col_idx];
        for (std::int64_t i = loc_idx; i < height_; i += range) {
            const Float* const inp_row = input_ + lstride_ * i;
            acc = binary_.native(acc, unary_(inp_row[col_idx]));
        }
        // WG reduction
        output_[col_idx] = sycl::reduce_over_group( //
            it.get_group(),
            acc,
            binary_.native);
    }

private:
    const Float* const input_;
    Float* const output_;
    const UnaryOp unary_;
    const BinaryOp binary_;
    const std::int64_t height_;
    const std::int32_t lstride_;
    const bool override_init_;
};

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::reduction_rm_cw_naive(sycl::queue& q,
                                                                       std::int64_t wg)
        : q_(q),
          wg_(wg) {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::reduction_rm_cw_naive(sycl::queue& q)
        : reduction_rm_cw_naive(q, propose_wg_size(q)) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
    ONEDAL_ASSERT(0 <= width && width <= stride);
    auto event = q_.submit([&, this](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(width);
        const auto kernel = get_kernel(input, output, height, stride, binary, unary, override_init);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    return this->
    operator()(input, output, width, height, width, binary, unary, deps, override_init);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::nd_range<2> reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::get_range(
    std::int64_t width) const {
    return make_multiple_nd_range_2d({ width, wg_ }, { 1, wg_ });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
typename reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::kernel_t
reduction_rm_cw_naive<Float, BinaryOp, UnaryOp>::get_kernel(const Float* input,
                                                            Float* output,
                                                            std::int64_t height,
                                                            std::int64_t stride,
                                                            const BinaryOp& binary,
                                                            const UnaryOp& unary,
                                                            const bool override_init) {
    return kernel_t{ input,
                     output,
                     dal::detail::integral_cast<std::int64_t>(height),
                     dal::detail::integral_cast<std::int32_t>(stride),
                     binary,
                     unary,
                     override_init };
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_naive<F, B, U>;

#define INSTANTIATE_FLOAT(B, U)                \
    INSTANTIATE(double, B<double>, U<double>); \
    INSTANTIATE(float, B<float>, U<float>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

INSTANTIATE_FLOAT(logical_or, isinfornan)
INSTANTIATE_FLOAT(logical_or, isinf)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
