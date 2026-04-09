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
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp, int f, int b>
class reduction_kernel {
    static_assert(f > 0);
    static_assert(b > 0);

public:
    constexpr static inline std::int32_t block = b;
    constexpr static inline std::int32_t folding = f;

    reduction_kernel(const Float* input,
                     Float* output,
                     std::int32_t width,
                     std::int64_t height,
                     std::int32_t lstride,
                     const BinaryOp& binary,
                     const UnaryOp& unary)
            : input_{ input },
              output_{ output },
              unary_{ unary },
              binary_{ binary },
              width_{ width },
              height_{ height },
              lstride_{ lstride } {}

    void operator()(sycl::nd_item<2> it) const {
        AccT accs[folding];
        for (std::int32_t i = 0; i < folding; ++i) {
            accs[i] = BinaryOp::init_value;
        }

        const std::int32_t vid = it.get_global_id(1);
        const std::int32_t hid = it.get_global_id(0);
        const std::int32_t hwg = it.get_global_range(0);
        for (std::int32_t i = 0; i < block; ++i) {
            const std::int64_t rid = vid * block + i;
            for (std::int32_t j = 0; j < folding; ++j) {
                const auto cid = hid + j * hwg;
                const auto* ptr = input_ + cid + rid * lstride_;
                const bool handle = (rid < height_) && (cid < width_);
                const auto bval = handle ? unary_(*ptr) : BinaryOp::init_value;
                accs[j] = binary_(accs[j], bval);
            }
        }

        for (std::int32_t j = 0; j < folding; ++j) {
            const auto cid = hid + j * hwg;
            if (cid < width_) {
                auto* const dst_ptr = output_ + cid;
                atomic_binary_op<BinaryOp>(dst_ptr, accs[j]);
            }
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
};

template <typename Float,
          typename AccT,
          typename BinaryOp,
          typename UnaryOp,
          int folding,
          int block_size>
sycl::event reduction_impl(sycl::queue& queue,
                           const Float* input,
                           Float* output,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           const BinaryOp& binary,
                           const UnaryOp& unary,
                           const event_vector& deps) {
    constexpr auto max_folding =
        reduction_rm_cw_atomic<Float, AccT, BinaryOp, UnaryOp>::max_folding;
    using kernel_t = reduction_kernel<Float, AccT, BinaryOp, UnaryOp, folding, block_size>;
    constexpr auto bl = kernel_t::block;
    const auto n_blocks = height / bl + bool(height % bl);
    const auto wg = std::min<std::int64_t>(device_max_wg_size(queue), width);
    const auto cfolding = width / wg + bool(width % wg);

    if (folding == cfolding) {
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            const auto range = make_multiple_nd_range_2d({ wg, n_blocks }, { wg, 1l });
            h.parallel_for<kernel_t>(range,
                                     kernel_t(input,
                                              output,
                                              dal::detail::integral_cast<std::int32_t>(width),
                                              dal::detail::integral_cast<std::int64_t>(height),
                                              dal::detail::integral_cast<std::int32_t>(stride),
                                              binary,
                                              unary));
        });
    }

    if constexpr ((max_folding >= folding) && (folding > 1)) {
        return reduction_impl<Float, AccT, BinaryOp, UnaryOp, folding - 1, block_size>(queue,
                                                                                       input,
                                                                                       output,
                                                                                       width,
                                                                                       stride,
                                                                                       height,
                                                                                       binary,
                                                                                       unary,
                                                                                       deps);
    }
    else {
        return reduction_rm_cw_naive<Float, AccT, BinaryOp, UnaryOp>{
            queue
        }(input, output, width, stride, height, binary, unary, deps);
    }
};

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_atomic<Float, AccT, BinaryOp, UnaryOp>::reduction_rm_cw_atomic(sycl::queue& q)
        : q_(q) {}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_atomic<Float, AccT, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    throw unimplemented(dal::detail::error_messages::method_not_implemented());

    /*
    TODO: This implementation contains a bug that prevents it from working correctly.
          It is commented out to avoid compilation errors, but it should be fixed in the future.
    event_vector new_deps{ deps };
    if (override_init) {
        auto view = ndview<Float, 1>::wrap(output, { width });
        new_deps.push_back(fill(q_, view, binary.init_value, deps));
    }
    auto res = reduction_impl<Float, BinaryOp, UnaryOp, max_folding, block_size>(q_,
                                                                                 input,
                                                                                 output,
                                                                                 width,
                                                                                 stride,
                                                                                 height,
                                                                                 binary,
                                                                                 unary,
                                                                                 new_deps);
    return res;
    */
}

template <typename Float, typename AccT, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_atomic<Float, AccT, BinaryOp, UnaryOp>::operator()(
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

#define INSTANTIATE(F, A, B, U) template class reduction_rm_cw_atomic<F, A, B, U>;

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
