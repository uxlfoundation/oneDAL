/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cstddef>
#include <type_traits>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"
#include "oneapi/dal/table/backend/convert/common_convert.hpp"

namespace oneapi::dal::backend {

template <bool mut, typename Pointer>
dal::array<Pointer> compute_pointers(const dal::array<dal::byte_t>& data,
                                     const dal::array<std::int64_t>& offsets) {
    const std::int64_t count = offsets.get_count();
    const std::int64_t* const raw_offsets = offsets.get_data();
    using ptr_t = std::conditional_t<mut, dal::byte_t*, const dal::byte_t*>;
    static_assert(std::is_same_v<Pointer, ptr_t>);

    ptr_t source = nullptr;
    if constexpr (mut) {
        source = data.get_mutable_data();
    }
    else {
        source = data.get_data();
    }

    auto pointers = dal::array<ptr_t>::empty(count);
    ptr_t* raw_pointers = pointers.get_mutable_data();

    PRAGMA_FORCE_SIMD
    for (std::int64_t row = 0l; row < count; ++row) {
        raw_pointers[row] = source + raw_offsets[row];
    }

    return pointers;
}

dal::array<std::int64_t> compute_output_offsets(data_type output_type,
                                                const shape_t& input_shape,
                                                const shape_t& output_strides) {
    const auto [row_count, col_count] = input_shape;
    const auto [row_stride, col_stride] = output_strides;
    const auto type_size = detail::get_data_type_size(output_type);
    const std::int64_t row_stride_in_bytes = type_size * row_stride;

    auto offsets = dal::array<std::int64_t>::empty(row_count);
    detail::check_mul_overflow(row_count, row_stride_in_bytes);
    std::int64_t* const raw_offsets = offsets.get_mutable_data();

    PRAGMA_FORCE_SIMD
    for (std::int64_t row = 0l; row < row_count; ++row) {
        raw_offsets[row] = row * row_stride_in_bytes;
    }

    return offsets;
}

std::int64_t align_offset(std::int64_t base, std::int64_t align) {
    using detail::check_sum_overflow;
    const std::int64_t residue = base % align;
    const std::int64_t diff = align - residue;
    const std::int64_t potential = check_sum_overflow(base, diff);
    const std::int64_t res = residue ? potential : base;
    ONEDAL_ASSERT(res % align == std::int64_t{ 0l });
    ONEDAL_ASSERT(res - base < align);
    return res;
}

dal::array<std::int64_t> compute_input_offsets(const shape_t& input_shape,
                                               const data_type* input_types) {
    ONEDAL_ASSERT(input_types != nullptr);
    const auto [row_count, col_count] = input_shape;
    ONEDAL_ASSERT((0l < row_count) && (0l < col_count));
    auto offsets = dal::array<std::int64_t>::empty(row_count);
    std::int64_t* const offsets_ptr = offsets.get_mutable_data();

    std::int64_t offset = 0l;
    for (std::int64_t row = 0l; row < row_count; ++row) {
        const data_type dtype = input_types[row];
        ONEDAL_ASSERT(is_known_data_type(dtype));

        auto raw_align = detail::get_data_type_align(dtype);

        offset = align_offset(offset, raw_align);
        offsets_ptr[row] = std::int64_t{ offset };

        auto raw_size = detail::get_data_type_size(dtype);
        auto row_size = detail::check_mul_overflow(raw_size, col_count);
        offset = detail::check_sum_overflow(offset, row_size);
    }

    return offsets;
}

dal::array<std::int64_t> compute_input_offsets(const shape_t& input_shape,
                                               const dal::array<data_type>& input_types) {
    return compute_input_offsets(input_shape, input_types.get_data());
}

template dal::array<dal::byte_t*> compute_pointers<true>(const dal::array<dal::byte_t>&,
                                                         const dal::array<std::int64_t>&);

template dal::array<const dal::byte_t*> compute_pointers<false>(const dal::array<dal::byte_t>&,
                                                                const dal::array<std::int64_t>&);

} // namespace oneapi::dal::backend
