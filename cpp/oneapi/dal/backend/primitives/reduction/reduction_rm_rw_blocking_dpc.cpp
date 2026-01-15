/*******************************************************************************
* Copyright 2024 Intel Corporation
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
namespace bk = dal::backend;
template <typename Float>
std::int64_t propose_block_size(const sycl::queue& q, const std::int64_t r) {
    constexpr std::int64_t fsize = sizeof(Float);
    return 0x10000l * (8 / fsize);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_rw_blocking<Float, BinaryOp, UnaryOp>::reduction_rm_rw_blocking(sycl::queue& q,
                                                                             std::int64_t wg)
        : q_(q),
          wg_(wg) {
    ONEDAL_ASSERT(0 < wg_ && wg_ <= device_max_wg_size(q_));
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_rw_blocking<Float, BinaryOp, UnaryOp>::reduction_rm_rw_blocking(sycl::queue& q)
        : reduction_rm_rw_blocking(q, propose_wg_size(q)) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw_blocking<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps,
    const bool override_init) const {
    const auto block_size = propose_block_size<Float>(q_, height);
    const bk::uniform_blocking blocking(height, block_size);
    std::vector<sycl::event> events(blocking.get_block_count());
    for (std::int64_t block_index = 0; block_index < blocking.get_block_count(); ++block_index) {
        const auto first_row = blocking.get_block_start_index(block_index);
        const auto last_row = blocking.get_block_end_index(block_index);
        const auto curr_block = last_row - first_row;
        ONEDAL_ASSERT(curr_block > 0);

        auto event = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(bk::make_multiple_nd_range_2d({ curr_block, wg_ }, { 1, wg_ }),
                             [=](sycl::nd_item<2> it) {
                                 // Common for whole WG
                                 const auto row_idx = it.get_global_id(0) + first_row;
                                 const auto loc_idx = it.get_local_id(1);
                                 const auto range = it.get_global_range(1);
                                 // It should be converted to upper type by default
                                 const Float* const inp_row = input + stride * row_idx;
                                 // Exclusive for EU
                                 Float acc = (override_init || (loc_idx != 0)) //
                                                 ? binary.init_value
                                                 : output[row_idx];
                                 for (std::int32_t i = loc_idx; i < width; i += range) {
                                     acc = binary.native(acc, unary(inp_row[i]));
                                 }
                                 // WG reduction
                                 output[row_idx] =
                                     sycl::reduce_over_group(it.get_group(), acc, binary.native);
                             });
        });

        events.push_back(event);
    }
    return bk::wait_or_pass(events);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_rw_blocking<Float, BinaryOp, UnaryOp>::operator()(
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

#define INSTANTIATE(F, B, U) template class reduction_rm_rw_blocking<F, B, U>;

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

} // namespace oneapi::dal::backend::primitives
