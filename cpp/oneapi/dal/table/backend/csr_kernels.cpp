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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/backend/convert.hpp"

#include <algorithm>

namespace oneapi::dal::backend {

using error_msg = dal::detail::error_messages;

ONEDAL_FORCEINLINE void check_origin_data(const array<byte_t>& origin_data,
                                          std::int64_t element_count,
                                          std::int64_t origin_dtype_size,
                                          std::int64_t block_dtype_size) {
    detail::check_mul_overflow(element_count, std::max(origin_dtype_size, block_dtype_size));
    ONEDAL_ASSERT(origin_data.get_count() >= element_count * origin_dtype_size);
}

std::int64_t csr_get_non_zero_count(const std::int64_t row_count,
                                    const std::int64_t* const row_offsets) {
    if (row_count <= 0)
        return 0;
    return row_offsets[row_count] - row_offsets[0];
}

std::int64_t csr_get_non_zero_count(detail::default_host_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* const row_offsets) {
    return csr_get_non_zero_count(row_count, row_offsets);
}

#ifdef ONEDAL_DATA_PARALLEL

std::int64_t csr_get_non_zero_count(sycl::queue& queue,
                                    const std::int64_t row_count,
                                    const std::int64_t* const row_offsets,
                                    const std::vector<sycl::event>& dependencies = {}) {
    if (row_count <= 0)
        return 0;

    if (!is_device_friendly_usm(queue, row_offsets))
        return csr_get_non_zero_count(row_count, row_offsets);

    std::int64_t first_row_offset{ 0L }, last_row_offset{ 0L };
    auto first_row_event = copy_usm2host(queue, &first_row_offset, row_offsets, 1, dependencies);
    auto last_row_event =
        copy_usm2host(queue, &last_row_offset, row_offsets + row_count, 1, dependencies);
    sycl::event::wait_and_throw({ first_row_event, last_row_event });

    return (last_row_offset - first_row_offset);
}

std::int64_t csr_get_non_zero_count(detail::data_parallel_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* const row_offsets,
                                    const std::vector<sycl::event>& dependencies = {}) {
    return csr_get_non_zero_count(policy.get_queue(), row_count, row_offsets, dependencies);
}

#endif

/// Provides access to the block of rows in `data` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
/// @tparam BlockData    The type of elements in the output array. In case the data values in CSR table
///                      are stored in the different data type, the conversion to `BlockData` type needs
///                      to be performed.
///                      The :literal:`BlockData` type should be at least :expr:`float`, :expr:`double`
///                      or :expr:`std::int32_t`.
///
/// @param[in] policy               Execution policy.
/// @param[in] origin_info          Information about this block of rows in the table.
///                                 Contains: layout, number of rows, information about data type,
///                                 indexing, etc.
/// @param[in] origin_data          Memory block that stores `data` array in the CSR table.
/// @param[in] same_data_type       True if the data type of the data in CSR table is the same as the data
///                                 type requested by the accessor's pull method. False otherwise.
/// @param[in] origin_offset        Index of the starting element of the `data` array in CSR table
///                                 to be pulled.
/// @param[in] block_size           Number of elemenst of the `data` array in CSR table to be pulled.
/// @param[out] data                Array that stores the block of rows of `data` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy, typename BlockData>
void pull_data_impl(const Policy& policy,
                    const csr_info& origin_info,
                    const array<byte_t>& origin_data,
                    const bool same_data_type,
                    const std::int64_t origin_offset,
                    const std::int64_t block_size,
                    array<BlockData>& data,
                    alloc_kind kind,
                    bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    if (same_data_type && !alloc_kind_requires_copy(get_alloc_kind(origin_data), kind)) {
        refer_origin_data(origin_data,
                          origin_offset * block_dtype_size,
                          block_size,
                          data,
                          preserve_mutability);
    }
    else {
        if (data.get_count() < block_size || !data.has_mutable_data() ||
            alloc_kind_requires_copy(get_alloc_kind(data), kind)) {
            reset_array(policy, data, block_size, kind);
        }

        const byte_t* const src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
        BlockData* const dst_data = data.get_mutable_data();

        backend::convert_vector(policy,
                                src_data,
                                dst_data,
                                origin_info.dtype_,
                                block_dtype,
                                block_size);
    }
}

/// Provides access to the block of rows in `column_indices` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
///
/// @param[in] policy                   Execution policy.
/// @param[in] origin_column_indices    `column_indices` array in the CSR table.
/// @param[in] origin_offset            Index of the starting element of the `column_indices` array
///                                     in CSR table to be pulled.
/// @param[in] block_size               Number of elemenst of the `column_indices` array in CSR table
///                                     to be pulled.
/// @param[in] indices_offset           The offset between the indices in the CSR table and the indices
///                                     requested by the pull method:
///                                         0, if the indexing is the same in the table
///                                            and in the pull method;
///                                         1, if the table has zero-based indexing
///                                            and pull method resuests one=based indexing;
///                                        -1, if the table has one-based indexing
///                                            and pull method resuests zero=based indexing.
/// @param[out] column_indices      Array that stores the block of rows of `column_indices` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy>
void pull_column_indices_impl(const Policy& policy,
                              const array<std::int64_t>& origin_column_indices,
                              const std::int64_t origin_offset,
                              const std::int64_t block_size,
                              const std::int64_t indices_offset,
                              array<std::int64_t>& column_indices,
                              alloc_kind kind,
                              bool preserve_mutability) {
    if (!alloc_kind_requires_copy(get_alloc_kind(origin_column_indices), kind) &&
        indices_offset == 0) {
        refer_origin_data(origin_column_indices,
                          origin_offset,
                          block_size,
                          column_indices,
                          preserve_mutability);
    }
    else {
        reset_array(policy, column_indices, block_size, kind);

        const std::int64_t* const src_data = origin_column_indices.get_data() + origin_offset;
        std::int64_t* const dst_data = column_indices.get_mutable_data();
#ifdef ONEDAL_DATA_PARALLEL
        if constexpr (detail::is_data_parallel_policy_v<Policy>) {
            auto copy_event =
                backend::copy_all2all(policy.get_queue(), dst_data, src_data, block_size);
            auto shift_event = shift_array_values(policy,
                                                  dst_data,
                                                  data_type::int64,
                                                  block_size,
                                                  &indices_offset,
                                                  { copy_event });
            sycl::event::wait_and_throw({ copy_event, shift_event });
        }
        else
#endif
        {
            backend::copy(dst_data, src_data, block_size);
            shift_array_values(policy, dst_data, data_type::int64, block_size, &indices_offset);
        }
    }
}

/// Provides access to the block of values in `row_offsets` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
///
/// @param[in] policy                   Execution policy.
/// @param[in] origin_row_offsets       `row_offsets` array in the CSR table.
/// @param[in] origin_offset            Index of the starting element of the `row_offsets` array
///                                     in CSR table to be pulled.
/// @param[in] block_size               Number of elemenst of the `row_offsets` array in CSR table
///                                     to be pulled.
/// @param[in] shift                    Zero-based index of the starting element of the `data` and `column_indices` arrays
///                                     in CSR table to be pulled.
/// @param[in] indices_offset           The offset between the indices in the CSR table and the indices
///                                     requested by the pull method:
///                                         0, if the indexing is the same in the table
///                                            and in the pull method;
///                                         1, if the table has zero-based indexing
///                                            and pull method resuests one=based indexing;
///                                        -1, if the table has one-based indexing
///                                            and pull method resuests zero=based indexing.
/// @param[out] row_offsets         Array that stores the block of values of `row_offsets` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy>
void pull_row_offsets_impl(const Policy& policy,
                           const array<std::int64_t>& origin_row_offsets,
                           const std::int64_t origin_offset,
                           const std::int64_t block_size,
                           const std::int64_t shift,
                           const std::int64_t indices_offset,
                           array<std::int64_t>& row_offsets,
                           alloc_kind kind,
                           bool preserve_mutability) {
    if (!alloc_kind_requires_copy(get_alloc_kind(origin_row_offsets), kind) && origin_offset == 0 &&
        indices_offset == 0) {
        refer_origin_data(origin_row_offsets, 0, block_size, row_offsets, preserve_mutability);
    }
    else {
        reset_array(policy, row_offsets, block_size, kind);

        const std::int64_t* const src_data = origin_row_offsets.get_data() + origin_offset;
        std::int64_t* const dst_data = row_offsets.get_mutable_data();

#ifdef ONEDAL_DATA_PARALLEL
        if constexpr (detail::is_data_parallel_policy_v<Policy>) {
            auto copy_event =
                backend::copy_all2all(policy.get_queue(), dst_data, src_data, block_size);
            auto shift_event = shift_array_values(policy,
                                                  dst_data,
                                                  data_type::int64,
                                                  block_size,
                                                  &shift,
                                                  { copy_event });
            sycl::event::wait_and_throw({ copy_event, shift_event });
        }
        else
#endif
        {
            backend::copy(dst_data, src_data, block_size);
            shift_array_values(policy, dst_data, data_type::int64, block_size, &shift);
        }
    }
}

template <typename Policy, typename BlockData>
void csr_pull_block(const Policy& policy,
                    const csr_info& origin_info,
                    const block_info& block_info,
                    const array<byte_t>& origin_data,
                    const array<std::int64_t>& origin_column_indices,
                    const array<std::int64_t>& origin_row_offsets,
                    array<BlockData>& data,
                    array<std::int64_t>& column_indices,
                    array<std::int64_t>& row_offsets,
                    alloc_kind kind,
                    bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    // overflows checked here
    check_origin_data(origin_data, origin_info.element_count_, origin_dtype_size, block_dtype_size);

    std::int64_t origin_offset = 0LL;
    std::int64_t block_size = 0LL;

    const bool same_data_type(block_dtype == origin_info.dtype_);

    const int indices_offset = (origin_info.indexing_ == block_info.indexing_)
                                   ? 0
                                   : ((origin_info.indexing_ == sparse_indexing::zero_based &&
                                       block_info.indexing_ == sparse_indexing::one_based)
                                          ? 1
                                          : -1);

    const auto* const origin_row_offsets_ptr = origin_row_offsets.get_data();
#ifdef ONEDAL_DATA_PARALLEL
    override_policy(policy, origin_row_offsets, row_offsets, [&](auto overriden_policy) {
        origin_offset = csr_get_non_zero_count(overriden_policy,
                                               block_info.row_offset_,
                                               origin_row_offsets_ptr);
        block_size = csr_get_non_zero_count(overriden_policy,
                                            block_info.row_count_,
                                            origin_row_offsets_ptr + block_info.row_offset_);
    });
#endif
    if (block_size == 0LL) {
        origin_offset = csr_get_non_zero_count(block_info.row_offset_, origin_row_offsets_ptr);
        block_size = csr_get_non_zero_count(block_info.row_count_,
                                            &origin_row_offsets_ptr[block_info.row_offset_]);
    }

    ONEDAL_ASSERT(origin_offset >= 0);
    ONEDAL_ASSERT(block_size >= 0);

    override_policy(policy, origin_data, data, [&](auto overriden_policy) {
        pull_data_impl(overriden_policy,
                       origin_info,
                       origin_data,
                       same_data_type,
                       origin_offset,
                       block_size,
                       data,
                       kind,
                       preserve_mutability);
    });

    override_policy(policy, origin_column_indices, column_indices, [&](auto overriden_policy) {
        pull_column_indices_impl(overriden_policy,
                                 origin_column_indices,
                                 origin_offset,
                                 block_size,
                                 indices_offset,
                                 column_indices,
                                 kind,
                                 preserve_mutability);
    });

    override_policy(policy, origin_row_offsets, row_offsets, [&](auto overriden_policy) {
        pull_row_offsets_impl(overriden_policy,
                              origin_row_offsets,
                              block_info.row_offset_,
                              block_info.row_count_ + 1,
                              indices_offset - origin_offset,
                              indices_offset,
                              row_offsets,
                              kind,
                              preserve_mutability);
    });
}

std::int64_t csr_get_non_zero_count(const array<std::int64_t>& row_offsets) {
    const std::int64_t row_count = row_offsets.get_count() - 1;
#ifdef ONEDAL_DATA_PARALLEL
    auto optional_queue = row_offsets.get_queue();
    if (optional_queue) {
        return csr_get_non_zero_count(optional_queue.value(),
                                      row_count,
                                      row_offsets.get_data(),
                                      {});
    }
#endif
    return csr_get_non_zero_count(row_count, row_offsets.get_data());
}

#ifdef ONEDAL_DATA_PARALLEL

/// Checks that the elements in the array allocated on device are not descending
///
/// @tparam T   The type of elements in the input array
///
/// @param[in,out] queue        The SYCL* queue object
/// @param[in]     count        The number of elements in the array
/// @param[in]     data         The pointer to the input array
/// @param[in]     dependencies Events indicating availability of the `data` for reading
///
/// @return true, if the elements in the array are not descending;
///         false, otherwise
template <typename T>
bool is_sorted(sycl::queue& queue,
               const std::int64_t count,
               const T* data,
               const std::vector<sycl::event>& dependencies) {
    // number of pairs of the subsequent elements in the data array that are sorted in desccending order,
    // i.e. for which data[i] > data[i + 1] is true.
    std::int64_t count_descending_pairs = 0L;

    sycl::buffer<std::int64_t, 1> count_buf(&count_descending_pairs, sycl::range<1>(1));

    const auto count_m1 = count - 1LL;
    const auto wg_size = dal::backend::device_max_wg_size(queue);
    const size_t count_m1_unsigned = static_cast<size_t>(count_m1);

    const size_t wg_count = (count_m1 + wg_size - 1) / wg_size;

    // count the number of pairs of the subsequent elements in the data array that are sorted
    // in desccending order using sycl::reduction
    queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(dependencies);
            auto count_descending_reduction =
                sycl::reduction(count_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());

            cgh.parallel_for(sycl::nd_range<1>{ wg_count * wg_size, wg_size },
                             count_descending_reduction,
                             [=](sycl::nd_item<1> idx, auto& count_descending) {
                                 const auto i = idx.get_global_id(0);
                                 if (i < count_m1_unsigned && data[i + 1] < data[i])
                                     count_descending.combine(1);
                             });
        })
        .wait_and_throw();

    return (count_descending_pairs == 0);
}

template <typename T>
bool is_sorted(const array<T>& arr, const std::vector<sycl::event>& dependencies) {
    const std::int64_t count = arr.get_count();
#ifdef ONEDAL_DATA_PARALLEL
    auto optional_queue = arr.get_queue();
    if (optional_queue) {
        return is_sorted(optional_queue.value(), count, arr.get_data(), dependencies);
    }
#endif
    const T* const data = arr.get_data();
    return std::is_sorted(data, data + count);
}

#endif

template <typename T>
out_of_bound_type check_bounds(const array<T>& arr, T min_value, T max_value) {
    const T* const data = arr.get_data();
    const auto count = arr.get_count();
    out_of_bound_type result{ out_of_bound_type::within_bounds };
    for (std::int64_t i = 0; i < count; ++i) {
        if (data[i] < min_value) {
            result = out_of_bound_type::less_than_min;
            break;
        }
        if (data[i] > max_value) {
            result = out_of_bound_type::greater_than_max;
            break;
        }
    }
    return result;
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
out_of_bound_type check_bounds(const array<T>& arr,
                               T min_value,
                               T max_value,
                               const std::vector<sycl::event>& dependencies) {
    auto optional_queue = arr.get_queue();
    if (!optional_queue)
        return check_bounds(arr, min_value, max_value);

    auto queue = optional_queue.value();
    const T* const data = arr.get_data();
    const auto count = arr.get_count();

    // number of elements in the data array that are less than [lt] the min_value,
    // number of elements in the data array that are greater than [gt] the max_value.
    std::int64_t count_lt_min = 0L, count_gt_max = 0L;
    sycl::buffer<std::int64_t, 1> count_lt_buf(&count_lt_min, sycl::range<1>(1));
    sycl::buffer<std::int64_t, 1> count_gt_buf(&count_gt_max, sycl::range<1>(1));

    // count the number of elements which are less than min_vaule and
    // the the number of elements which are greater than max_value using sycl::reduction
    queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(dependencies);
            auto count_lt_reduction =
                sycl::reduction(count_lt_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());
            auto count_gt_reduction =
                sycl::reduction(count_gt_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());

            cgh.parallel_for(sycl::range<1>{ dal::detail::integral_cast<std::size_t>(count) },
                             count_lt_reduction,
                             count_gt_reduction,
                             [=](sycl::id<1> i, auto& count_lt, auto& count_gt) {
                                 if (data[i] < min_value) {
                                     count_lt.combine(1);
                                 }
                                 if (data[i] > max_value) {
                                     count_gt.combine(1);
                                 }
                             });
        })
        .wait_and_throw();

    out_of_bound_type result{ out_of_bound_type::within_bounds };
    if (count_lt_min > 0)
        result = out_of_bound_type::less_than_min;
    else if (count_gt_max > 0)
        result = out_of_bound_type::greater_than_max;
    return result;
}

#endif

#define INSTANTIATE(Policy, BlockData)                                             \
    template void csr_pull_block(const Policy& policy,                             \
                                 const csr_info& origin_info,                      \
                                 const block_info& block_info,                     \
                                 const array<byte_t>& origin_data,                 \
                                 const array<std::int64_t>& origin_column_indices, \
                                 const array<std::int64_t>& origin_row_offsets,    \
                                 array<BlockData>& data,                           \
                                 array<std::int64_t>& column_indices,              \
                                 array<std::int64_t>& row_offsets,                 \
                                 alloc_kind requested_alloc_kind,                  \
                                 bool preserve_mutability);

#define INSTANTIATE_HOST_POLICY(Data) INSTANTIATE(detail::default_host_policy, Data)
#define INSTANTIATE_DP_POLICY(Data)   INSTANTIATE(detail::data_parallel_policy, Data)

INSTANTIATE_HOST_POLICY(float)
INSTANTIATE_HOST_POLICY(double)
INSTANTIATE_HOST_POLICY(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL
INSTANTIATE_DP_POLICY(float)
INSTANTIATE_DP_POLICY(double)
INSTANTIATE_DP_POLICY(std::int32_t)
#endif

template out_of_bound_type check_bounds(const array<std::int64_t>& arr,
                                        std::int64_t min_value,
                                        std::int64_t max_value);

#ifdef ONEDAL_DATA_PARALLEL
template bool is_sorted(const array<std::int64_t>& arr,
                        const std::vector<sycl::event>& dependencies);
template out_of_bound_type check_bounds(const array<std::int64_t>& arr,
                                        std::int64_t min_value,
                                        std::int64_t max_value,
                                        const std::vector<sycl::event>& dependencies);

#endif

} // namespace oneapi::dal::backend
