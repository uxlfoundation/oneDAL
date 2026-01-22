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
class kernel_reduction_rm_rw_narrow {
public:
    kernel_reduction_rm_rw_narrow(const Float* input,
                                  Float* output,
                                  std::int32_t width,
                                  std::int64_t height,
                                  std::int32_t lstride,
                                  const BinaryOp& binary,
                                  const UnaryOp& unary,
                                  const bool override_init)
            : input_{ input },
              output_{ output },
              unary_{ unary },
              binary_{ binary },
              width_{ width },
              height_{ height },
              lstride_{ lstride },
              override_init_{ override_init } {}
    void operator()(sycl::nd_item<1> it) const {
        const std::int64_t row_idx = it.get_global_id(0);
        if (row_idx < height_) {
            // It should be converted to upper type by default
            const Float* const inp_row = input_ + lstride_ * row_idx;
            // Exclusive for the EU
            AccT acc = override_init_ ? binary_.init_value : output_[row_idx];
            for (std::int32_t i = 0; i < width_; ++i) {
                acc = binary_.native(acc, unary_(inp_row[i]));
            }
            output_[row_idx] = acc;
        }
    }

private:
    const Float* const input_;
    Float* const output_;
    const UnaryOp unary_;
    const BinaryOp binary_;
    const std::int32_t width_;
    const std::int64_t height_;
    const std::int32_t lstride_;
    const bool override_init_;
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::reduction_rm_rw_narrow(sycl::queue& q,
                                                                               std::int64_t wg)
        : q_(q),
          wg_(wg) {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::reduction_rm_rw_narrow(sycl::queue& q)
        : reduction_rm_rw_narrow(q, propose_wg_size(q)) {}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    auto event = q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = get_range(height);
        const auto kernel =
            get_kernel(input, output, width, height, stride, binary, unary, override_init);
        h.parallel_for<kernel_t>(range, kernel);
    });
    return event;
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::operator()(
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

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::nd_range<1> reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::get_range(
    std::int64_t height) const {
    const auto nblocks = (height / wg_) + bool(height % wg_);
    const auto eheight = wg_ * nblocks;
    ONEDAL_ASSERT(height <= eheight);
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
    return make_multiple_nd_range_1d(eheight, wg_);
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
typename reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::kernel_t
reduction_rm_rw_narrow<Float, AccT, BinaryOp, UnaryOp>::get_kernel(const Float* input,
                                                                   Float* output,
                                                                   std::int64_t width,
                                                                   std::int64_t height,
                                                                   std::int64_t stride,
                                                                   const BinaryOp& binary,
                                                                   const UnaryOp& unary,
                                                                   const bool override_init) {
    ONEDAL_ASSERT(0 <= width && width <= stride);
    return kernel_t{ input,
                     output,
                     dal::detail::integral_cast<std::int32_t>(width),
                     dal::detail::integral_cast<std::int64_t>(height),
                     dal::detail::integral_cast<std::int32_t>(stride),
                     binary,
                     unary,
                     override_init };
}

#define INSTANTIATE(F, A, B, U) template class reduction_rm_rw_narrow<F, A, B, U>;

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
