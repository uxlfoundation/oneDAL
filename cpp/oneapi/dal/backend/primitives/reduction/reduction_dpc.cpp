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

#include <type_traits>

#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw_dpc.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename BinaryOp, typename UnaryOp>
inline sycl::event reduce_rm_rw(sycl::queue& q,
                                const ndview<Float, 2, ndorder::c>& input,
                                ndview<Float, 1>& output,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps,
                                bool override_init = true) {
    ONEDAL_ASSERT(input.has_data());
    ONEDAL_ASSERT(output.has_mutable_data());
    ONEDAL_ASSERT(0 <= input.get_dimension(1));
    ONEDAL_ASSERT(0 <= input.get_dimension(0));
    ONEDAL_ASSERT(input.get_dimension(0) <= output.get_dimension(0));
    using kernel_t = reduction_rm_rw<Float, BinaryOp, UnaryOp>;
    const auto width = input.get_dimension(1);
    const auto height = input.get_dimension(0);
    const auto stride = input.get_leading_stride();
    const auto* inp_ptr = input.get_data();
    auto* out_ptr = output.get_mutable_data();
    const kernel_t kernel(q);
    return kernel(inp_ptr, out_ptr, width, height, stride, binary, unary, deps, override_init);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
inline sycl::event reduce_rm_cw(sycl::queue& q,
                                const ndview<Float, 2, ndorder::c>& input,
                                ndview<Float, 1>& output,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps,
                                bool override_init = true) {
    ONEDAL_ASSERT(input.has_data());
    ONEDAL_ASSERT(output.has_mutable_data());
    ONEDAL_ASSERT(0 <= input.get_dimension(1));
    ONEDAL_ASSERT(0 <= input.get_dimension(0));
    ONEDAL_ASSERT(input.get_dimension(1) <= output.get_dimension(0));
    using kernel_t = reduction_rm_cw<Float, BinaryOp, UnaryOp>;
    const auto width = input.get_dimension(1);
    const auto height = input.get_dimension(0);
    const auto stride = input.get_leading_stride();
    const auto* inp_ptr = input.get_data();
    auto* out_ptr = output.get_mutable_data();
    const kernel_t kernel(q);
    return kernel(inp_ptr, out_ptr, width, height, stride, binary, unary, deps, override_init);
}

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_by_rows_impl(sycl::queue& q,
                                const ndview<Float, 2, order>& input,
                                ndview<Float, 1>& output,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps,
                                bool override_init) {
    ONEDAL_ASSERT(input.get_dimension(0) <= output.get_dimension(0));
    if constexpr (order == ndorder::c) {
        return reduce_rm_rw(q, input, output, binary, unary, deps, override_init);
    }
    else {
        auto input_tr = input.t();
        return reduce_rm_cw(q, input_tr, output, binary, unary, deps, override_init);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

/// Reduces CSR table with `n x m` dimensions by rows
///
/// @tparam Float       Floating point type, it can be `float` or `double`
/// @tparam BinaryOp    Binary operation class, it reduces 2 input values into 1
/// @tparam UnaryOp     Unary operation class, it modifies an input value
///
/// @param[in] q                Sycl queue
/// @param[in] values           An array of values in CSR table
/// @param[in] column_indices   An array of column indices in CSR table
/// @param[in] row_offsets      An array of row offsets in CSR table
/// @param[in] indexing         Indexing kind of CSR table
/// @param[out] output          An output array with dimensions `n x 1`
/// @param[in] binary           A binary operation used in reduction
/// @param[in] unary            An unary operation used in reduction
/// @param[in] deps             A vector of dependent events
template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduce_by_rows_impl(sycl::queue& q,
                                const ndview<Float, 1>& values,
                                const ndview<std::int64_t, 1>& column_indices,
                                const ndview<std::int64_t, 1>& row_offsets,
                                const dal::sparse_indexing indexing,
                                ndview<Float, 1>& output,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps,
                                bool override_init) {
    ONEDAL_ASSERT(values.get_count() == column_indices.get_count());
    const std::int64_t row_block_size = device_max_wg_size(q);
    const std::int64_t column_block_size = device_max_wg_size(q) / 2;
    const auto range =
        make_multiple_nd_range_2d({ row_block_size, column_block_size }, { 1, column_block_size });
    const auto val_ptr = values.get_data();
    const auto row_ptr = row_offsets.get_data();
    auto const out_ptr = output.get_mutable_data();
    const std::int64_t shift = bool(indexing == sparse_indexing::one_based);
    const auto row_count = row_offsets.get_count() - 1;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](auto it) {
            const std::int64_t row_shift = it.get_global_id(0);
            const std::int64_t col_shift = it.get_local_id(1);
            for (auto row_idx = row_shift; row_idx < row_count; row_idx += row_block_size) {
                const auto start = row_ptr[row_idx] - shift;
                const auto end = row_ptr[row_idx + 1] - shift;
                Float local_accum = binary.init_value;
                for (auto idx = start + col_shift; idx < end; idx += column_block_size) {
                    const auto val = val_ptr[idx];
                    local_accum = binary.native(local_accum, unary(val));
                }
                const auto result =
                    sycl::reduce_over_group(it.get_group(), local_accum, binary.native);
                if (col_shift == 0) {
                    out_ptr[row_idx] = override_init ? result : out_ptr[row_idx] + result;
                }
            }
        });
    });
}

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_by_columns_impl(sycl::queue& q,
                                   const ndview<Float, 2, order>& input,
                                   ndview<Float, 1>& output,
                                   const BinaryOp& binary,
                                   const UnaryOp& unary,
                                   const event_vector& deps,
                                   bool override_init) {
    ONEDAL_ASSERT(input.get_dimension(1) <= output.get_dimension(0));
    if constexpr (order == ndorder::c) {
        return reduce_rm_cw(q, input, output, binary, unary, deps, override_init);
    }
    else {
        auto input_tr = input.t();
        return reduce_rm_rw(q, input_tr, output, binary, unary, deps, override_init);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

#define INSTANTIATE(F, L, B, U)                                                     \
    template sycl::event reduce_by_rows_impl<F, L, B, U>(sycl::queue&,              \
                                                         const ndview<F, 2, L>&,    \
                                                         ndview<F, 1>&,             \
                                                         const B&,                  \
                                                         const U&,                  \
                                                         const event_vector&,       \
                                                         bool);                     \
    template sycl::event reduce_by_columns_impl<F, L, B, U>(sycl::queue&,           \
                                                            const ndview<F, 2, L>&, \
                                                            ndview<F, 1>&,          \
                                                            const B&,               \
                                                            const U&,               \
                                                            const event_vector&,    \
                                                            bool);
#define INSTANTIATE_CSR(F, B, U)                                                      \
    template sycl::event reduce_by_rows_impl<F, B, U>(sycl::queue&,                   \
                                                      const ndview<F, 1>&,            \
                                                      const ndview<std::int64_t, 1>&, \
                                                      const ndview<std::int64_t, 1>&, \
                                                      dal::sparse_indexing,           \
                                                      ndview<F, 1>&,                  \
                                                      const B&,                       \
                                                      const U&,                       \
                                                      const event_vector&,            \
                                                      bool);

#define INSTANTIATE_LAYOUT(F, B, U)  \
    INSTANTIATE(F, ndorder::c, B, U) \
    INSTANTIATE(F, ndorder::f, B, U) \
    INSTANTIATE_CSR(F, B, U)

#define INSTANTIATE_FLOAT(B, U)                       \
    INSTANTIATE_LAYOUT(double, B<double>, U<double>); \
    INSTANTIATE_LAYOUT(float, B<float>, U<float>);

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

#undef INSTANTIATE_LAYOUT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
