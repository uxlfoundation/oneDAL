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

#include <numeric>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/detail/debug.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/threading.hpp"

#include "oneapi/dal/detail/array_utils.hpp"
#include "oneapi/dal/detail/chunked_array_utils.hpp"

#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/heterogen_kernels.hpp"

#include "oneapi/dal/table/backend/convert/copy_convert.hpp"
#include "oneapi/dal/table/backend/convert/common_convert.hpp"

namespace oneapi::dal::backend {

using detail::operator<<;

template <typename Meta, typename Data>
inline std::int64_t get_column_count(const Meta& meta, const Data& data) {
    const auto result = meta.get_feature_count();
#ifdef ONEDAL_ENABLE_ASSERT
    ONEDAL_ASSERT(result == data.get_count());
#endif // ONEDAL_ENABLE_ASSERT
    return result;
}

template <typename Meta, typename Data>
inline std::int64_t get_row_count(std::int64_t column_count, const Meta& meta, const Data& data) {
    auto get_count = [&](std::int64_t c) -> std::int64_t {
        const auto& column = data[c];
        const auto dtype = meta.get_data_type(c);
        return detail::get_element_count(dtype, column);
    };

    ONEDAL_ASSERT(0l < column_count);
    const auto result = get_count(0l);

#ifdef ONEDAL_ENABLE_ASSERT
    for (std::int64_t c = 1l; c < column_count; ++c) {
        ONEDAL_ASSERT(get_count(c) == result);
    }
#endif // ONEDAL_ENABLE_ASSERT

    return result;
}

std::int64_t heterogen_column_count(const table_metadata& meta,
                                    const heterogen_data& data) {
    return get_column_count(meta, data);
}

std::int64_t heterogen_row_count(std::int64_t column_count,
                                 const table_metadata& meta,
                                 const heterogen_data& data) {
    return get_row_count(column_count, meta, data);
}

std::pair<std::int64_t, std::int64_t> heterogen_shape(const table_metadata& meta,
                                      const heterogen_data& data) {
    const auto column_count = heterogen_column_count(meta, data);
    const auto row_count = heterogen_row_count(column_count, meta, data);
    return std::pair<std::int64_t, std::int64_t>{ row_count, column_count };
}

std::int64_t heterogen_row_count(const table_metadata& meta,
                                 const heterogen_data& data) {
    return heterogen_shape(meta, data).first;
}

template <typename Policy>
struct heterogen_dispatcher {};

template <typename Meta, typename Data>
std::int64_t get_row_size(const Meta& meta, const Data& data) {
    std::int64_t acc = 0l;

    const auto col_count = get_column_count(meta, data);
    for (std::int64_t col = 0l; col < col_count; ++col) {
        const auto dtype = meta.get_data_type(col);
        acc += detail::get_data_type_size(dtype);
    }

    return acc;
}

template <typename Meta, typename Data>
std::int64_t propose_row_block_size(const Meta& meta, const Data& data) {
    constexpr std::int64_t estimation = 10'000'000;
    const auto row_size = get_row_size(meta, data);
    const auto col_count = get_column_count(meta, data);
    const auto row_count = get_row_count(col_count, meta, data);
    return std::max(1l, std::min(row_count, estimation / row_size));
}

heterogen_data heterogen_row_slice(const range& rows_range,
                                   const table_metadata& meta,
                                   const heterogen_data& data) {
    const auto col_count = get_column_count(meta, data);
    const auto row_count = get_row_count(col_count, meta, data);

    const auto rows = rows_range.normalize_range(row_count);
    const auto first = rows.start_idx, last = rows.end_idx;

    const detail::chunked_array_base empty{};
    auto result = heterogen_data::full(col_count, empty);
    auto* const res_ptr = result.get_mutable_data();

    detail::threader_for_int64(col_count, [&](std::int64_t col) {
        const auto dtype = meta.get_data_type(col);
        const auto elem_size = detail::get_data_type_size(dtype);

        const auto last_byte = detail::check_mul_overflow(elem_size, last);
        const auto first_byte = detail::check_mul_overflow(elem_size, first);

        auto column = chunked_array<dal::byte_t>::make(data[col]);
        auto slice = column.get_slice(first_byte, last_byte);

        res_ptr[col] = std::move(slice);
    });

    return result;
}

using sliced_buffer = array<array<dal::byte_t>>;

sliced_buffer slice_buffer(const shape_t& buff_shape,
                           const array<dal::byte_t>& buff,
                           const array<data_type>& data_types) {

    const auto [row_count, col_count] = buff_shape;

    const auto* const data_types_ptr = data_types.get_data();

    const dal::array<dal::byte_t> empty{};
    auto result = sliced_buffer::full(col_count, empty);
    auto* const result_ptr = result.get_mutable_data();

    std::int64_t first = 0l;
    for (std::int64_t c = 0l; c < col_count; ++c) {
        const data_type dtype = data_types_ptr[c];
        const auto dsize = detail::get_data_type_size(dtype);
        const auto csize = detail::check_mul_overflow(dsize, row_count);

        const auto last = detail::check_sum_overflow(first, csize);

        result_ptr[c] = buff.get_slice(first, last);

        first = last;
    }

    return result;
}

template <typename Policy>
void copy_buffer(const Policy& policy,
                 std::int64_t rows_count,
                 const heterogen_data& sliced_data,
                 sliced_buffer& sliced_buffer,
                 const array<data_type>& data_types) {
    const auto cols_count = sliced_buffer.get_count();
    ONEDAL_ASSERT(cols_count == sliced_data.get_count());

    const data_type* const dtypes = data_types.get_data();
    array<dal::byte_t>* columns = sliced_buffer.get_mutable_data();
    detail::threader_for_int64(cols_count, [&](std::int64_t col) {
        const auto element_size = detail::get_data_type_size(dtypes[col]);
        auto size = detail::check_mul_overflow(element_size, rows_count);
        auto col_slice = columns[col].get_slice(0l, size);
        detail::copy(col_slice, sliced_data[col]);
    });
}

template <>
struct heterogen_dispatcher<detail::host_policy> {
    using policy_t = detail::host_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const heterogen_data& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {
        const auto col_count = get_column_count(meta, data);
        const auto row_count = get_row_count(col_count, meta, data);
        const auto [first, last] = rows_range.normalize_range(row_count);

        ONEDAL_ASSERT(first < last);
        const auto copy_count = last - first;

        const auto row_size = get_row_size(meta, data);
        const auto block = propose_row_block_size(meta, data);
        const auto block_size = detail::check_mul_overflow(block, row_size);

        const auto& data_types = meta.get_data_types();
        auto buff = array<dal::byte_t>::empty(block_size);
        auto buff_shape = std::make_pair(block, col_count);
        auto buff_cols = slice_buffer(buff_shape, buff, data_types);

        const shape_t transposed_shape = transpose(buff_shape);
        auto buff_offs = compute_input_offsets(transposed_shape, data_types);
        auto buff_ptrs = compute_pointers</*mut=*/false>(buff, buff_offs);

        constexpr Type fill_value = std::numeric_limits<Type>::max();
        auto result_count = detail::check_mul_overflow(copy_count, col_count);
        auto result = dal::array<Type>::full(result_count, fill_value);

        constexpr auto type = detail::make_data_type<Type>();
        auto outp_types = array<data_type>::full(col_count, type);
        auto outp_strs = array<std::int64_t>::full(col_count, col_count);
        auto buff_strs = array<std::int64_t>::full(col_count, std::int64_t(1l));
        const shape_t transposed_strides = std::make_pair(1l, col_count);
        auto outp_offs = compute_output_offsets(type, //
                                    transposed_shape, transposed_strides);

        const auto block_count = (copy_count / block) + bool(copy_count % block);

        for (std::int64_t b = 0l; b < block_count; ++b) {
            auto start = detail::check_mul_overflow(b, block);
            const auto f = detail::check_sum_overflow(start, first);
            auto end = detail::check_sum_overflow(f, block);
            const auto l = std::min(end, last);
            const std::int64_t len = l - f;
            ONEDAL_ASSERT(len <= block);
            ONEDAL_ASSERT(0l < len);

            auto slice = heterogen_row_slice({f, l}, meta, data);
            copy_buffer(policy, len, slice, buff_cols, data_types);

            auto out_first = detail::check_mul_overflow(b, col_count);
            auto out_block = detail::check_mul_overflow(len, col_count);
            auto out_last = detail::check_sum_overflow(out_first, out_block);

            auto curr_shape = std::make_pair(len, col_count);
            auto out_slice = result.get_slice(out_first, out_last);
            auto raw_slice = array<dal::byte_t>(out_slice, //
                reinterpret_cast<dal::byte_t*>(out_slice.get_mutable_data()),
                detail::check_mul_overflow<std::int64_t>(len, sizeof(Type)));
            auto outp_ptrs = compute_pointers</*mut=*/true>(raw_slice, outp_offs);
            copy_convert(policy, buff_ptrs, data_types, buff_strs, outp_ptrs,
                                    outp_types, outp_strs, transpose(curr_shape));
        }

        block_data.reset(result, result.get_mutable_data(), result.get_count());
    }
};

template <>
struct heterogen_dispatcher<detail::default_host_policy> {
    using policy_t = detail::default_host_policy;

    template <typename Type>
    static void pull(const policy_t& default_policy,
                     const table_metadata& meta,
                     const heterogen_data& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind alloc_kind) {
        const auto policy = detail::host_policy::get_default();
        heterogen_dispatcher<detail::host_policy>::pull<Type>(
            policy, meta, data, block_data, rows_range, alloc_kind);
    }
};

#ifdef ONEDAL_DATA_PARALLEL

template <typename Queue, typename Meta, typename Data>
std::int64_t propose_row_block_size(const Queue& queue,
                    const Meta& meta, const Data& data) {
    return propose_row_block_size(meta, data);
}

template <>
struct heterogen_dispatcher<detail::data_parallel_policy> {
    using policy_t = detail::data_parallel_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const heterogen_data& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {
        sycl::queue& queue = policy.get_queue();
        auto host_policy = detail::host_policy::get_default();
        const auto alloc = alloc_kind_to_sycl(requested_alloc_kind);

        const auto col_count = get_column_count(meta, data);
        const auto row_count = get_row_count(col_count, meta, data);
        const auto [first, last] = rows_range.normalize_range(row_count);

        ONEDAL_ASSERT(first < last);
        const auto copy_count = last - first;

        const auto row_size = get_row_size(meta, data);
        const auto block = propose_row_block_size(queue, meta, data);
        const auto block_size = detail::check_mul_overflow(block, row_size);

        const auto& data_types = meta.get_data_types();
        auto buff_shape = std::make_pair(block, col_count);
        auto buff = array<dal::byte_t>::empty(queue, block_size, alloc);

        auto buff_cols = slice_buffer(buff_shape, buff, data_types);

        const shape_t transposed_shape = transpose(buff_shape);
        auto buff_offs = compute_input_offsets(transposed_shape, data_types);
        auto buff_ptrs = compute_pointers</*mut=*/false>(buff, buff_offs);

        constexpr Type fill_value = std::numeric_limits<Type>::max();
        auto result_count = detail::check_mul_overflow(copy_count, col_count);
        auto result = dal::array<Type>::full(queue, result_count, fill_value, alloc);

        constexpr auto type = detail::make_data_type<Type>();
        auto outp_types = array<data_type>::full(col_count, type);
        auto outp_strs = array<std::int64_t>::full(col_count, col_count);
        auto buff_strs = array<std::int64_t>::full(col_count, std::int64_t(1l));
        const shape_t transposed_strides = std::make_pair(1l, col_count);
        auto outp_offs = compute_output_offsets(type, //
                                    transposed_shape, transposed_strides);

        const auto block_count = (copy_count / block) + bool(copy_count % block);

        for (std::int64_t b = 0l; b < block_count; ++b) {
            auto start = detail::check_mul_overflow(b, block);
            const auto f = detail::check_sum_overflow(start, first);
            auto end = detail::check_sum_overflow(f, block);
            const auto l = std::min(end, last);
            const std::int64_t len = l - f;
            ONEDAL_ASSERT(len <= block);
            ONEDAL_ASSERT(0l < len);

            auto slice = heterogen_row_slice({f, l}, meta, data);
            copy_buffer(policy, len, slice, buff_cols, data_types);

            auto out_first = detail::check_mul_overflow(b, col_count);
            auto out_block = detail::check_mul_overflow(len, col_count);
            auto out_last = detail::check_sum_overflow(out_first, out_block);

            auto curr_shape = std::make_pair(len, col_count);
            auto out_slice = result.get_slice(out_first, out_last);
            auto raw_slice = array<dal::byte_t>(out_slice, //
                reinterpret_cast<dal::byte_t*>(out_slice.get_mutable_data()),
                detail::check_mul_overflow<std::int64_t>(len, sizeof(Type)));
            auto outp_ptrs = compute_pointers</*mut=*/true>(raw_slice, outp_offs);
            copy_convert(policy, buff_ptrs, data_types, buff_strs, outp_ptrs,
                                    outp_types, outp_strs, transpose(curr_shape));
        }

        block_data.reset(result, result.get_mutable_data(), result.get_count());
    }
};

#endif // ONEDAL_DATA_PARALLEL

template <typename Policy, typename Type>
void heterogen_pull_rows(const Policy& policy,
                         const table_metadata& meta,
                         const heterogen_data& data,
                         array<Type>& block_data,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind) {
    heterogen_dispatcher<Policy>::template pull<Type>(policy, meta, data,
        	                block_data, rows_range, requested_alloc_kind);
}

#define INSTANTIATE(Policy, Type)                                               \
    template void heterogen_pull_rows(const Policy&,                            \
                                      const table_metadata&,                    \
                                      const heterogen_data&,                    \
                                      array<Type>&,                             \
                                      const range&,                             \
                                      alloc_kind);

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE_ALL_POLICIES(Type)             \
    INSTANTIATE(detail::default_host_policy, Type) \
    INSTANTIATE(detail::data_parallel_policy, Type)
#else
#define INSTANTIATE_ALL_POLICIES(Type) INSTANTIATE(detail::default_host_policy, Type)
#endif

INSTANTIATE_ALL_POLICIES(float)
INSTANTIATE_ALL_POLICIES(double)
INSTANTIATE_ALL_POLICIES(std::int32_t)

} // namespace oneapi::dal::backend