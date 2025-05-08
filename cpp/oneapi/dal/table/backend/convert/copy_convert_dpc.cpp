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

#include "oneapi/dal/detail/array_utils.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"
#include "oneapi/dal/table/backend/convert/copy_convert.hpp"
#include "oneapi/dal/table/backend/convert/common_convert.hpp"
#include "oneapi/dal/table/backend/convert/copy_convert_impl.hpp"

namespace oneapi::dal::backend {

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::array<data_type>& input_types,
                         const dal::array<dal::byte_t>& input_data,
                         const shape_t& input_shape,
                         data_type output_type,
                         dal::array<dal::byte_t>& output_data,
                         const shape_t& output_strides,
                         const std::vector<sycl::event>& deps) {
    const auto [row_count, col_count] = input_shape;
    const auto [row_stride, col_stride] = output_strides;

    auto input_offsets = compute_input_offsets(input_shape, input_types);
    auto output_offsets = compute_output_offsets(output_type, input_shape, output_strides);

    auto input_pointers = compute_pointers<false>(input_data, input_offsets);
    auto output_pointers = compute_pointers<true>(output_data, output_offsets);

    auto output_types = dal::array<data_type>::full(row_count, output_type);
    auto output_strides_arr = dal::array<std::int64_t>::full(row_count, col_stride);
    auto input_strides = dal::array<std::int64_t>::full(row_count, std::int64_t{ 1l });

    return copy_convert(policy,
                        input_pointers,
                        input_types,
                        input_strides,
                        output_pointers,
                        output_types,
                        output_strides_arr,
                        input_shape,
                        deps);
}

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::array<const dal::byte_t*>& inp_pointers,
                         const dal::array<data_type>& inp_types,
                         const dal::array<std::int64_t>& inp_strides,
                         const dal::array<dal::byte_t*>& out_pointers,
                         const dal::array<data_type>& out_types,
                         const dal::array<std::int64_t>& out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    return copy_convert(policy,
                        inp_pointers.get_data(),
                        inp_types.get_data(),
                        inp_strides.get_data(),
                        out_pointers.get_data(),
                        out_types.get_data(),
                        out_strides.get_data(),
                        shape,
                        deps);
}

template <typename Input, typename Output>
struct single_row_info {
    Input* const inp_ptr;
    std::int64_t inp_str;
    Output* const out_ptr;
    std::int64_t out_str;
};

sycl::event copy_convert_one(const detail::data_parallel_policy& policy,
                             const dal::byte_t* const inp_pointer,
                             data_type inp_type,
                             std::int64_t inp_stride,
                             dal::byte_t* const out_pointer,
                             data_type out_type,
                             std::int64_t out_stride,
                             std::int64_t count,
                             const std::vector<sycl::event>& deps) {
    return copy_convert(policy,
                        &inp_pointer,
                        &inp_type,
                        &inp_stride,
                        &out_pointer,
                        &out_type,
                        &out_stride,
                        { 1l, count },
                        deps);
}

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    return copy_convert(policy.get_queue(),
                        inp_pointers,
                        inp_types,
                        inp_strides,
                        out_pointers,
                        out_types,
                        out_strides,
                        shape,
                        deps);
}

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    sycl::event::wait_and_throw(deps);
    const auto [row_count, col_count] = shape;

    constexpr auto device = sycl::usm::alloc::device;

    const dal::array<std::int64_t> unique_indices = //
        find_sets_of_unique_pairs(inp_types, out_types, row_count);
    const std::int64_t* const unique_indices_ptr = unique_indices.get_data();

    const dal::array<std::int64_t> chunk_offsets = //
        find_unique_chunk_offsets(unique_indices, inp_types, out_types);

    const auto inp_strides_host = //
        extract_by_indices(unique_indices_ptr, inp_strides, row_count);
    auto inp_strides_device = array<std::int64_t>::empty(queue, row_count, device);
    /* Copying to device */ detail::copy(inp_strides_device, inp_strides_host);

    const auto out_strides_host = //
        extract_by_indices(unique_indices_ptr, out_strides, row_count);
    auto out_strides_device = array<std::int64_t>::empty(queue, row_count, device);
    /* Copying to device */ detail::copy(out_strides_device, out_strides_host);

    const auto inp_pointers_host = //
        extract_by_indices(unique_indices_ptr, inp_pointers, row_count);
    auto inp_pointers_device = array<const dal::byte_t*>::empty(queue, row_count, device);
    /* Copying to device */ detail::copy(inp_pointers_device, inp_pointers_host);

    const auto out_pointers_host = //
        extract_by_indices(unique_indices_ptr, out_pointers, row_count);
    auto out_pointers_device = array<dal::byte_t*>::empty(queue, row_count, device);
    /* Copying to device */ detail::copy(out_pointers_device, out_pointers_host);

    std::int64_t first = 0l;
    sycl::event last_event{};

    const std::int64_t chunk_count = chunk_offsets.get_count();
    for (std::int64_t chunk = 0l; chunk < chunk_count; ++chunk) {
        const auto last = chunk_offsets[chunk];
        const auto chunk_len = last - first;
        ONEDAL_ASSERT(0l < chunk_len);

        const auto first_idx = unique_indices_ptr[first];
        const data_type inp_type = inp_types[first_idx];
        const data_type out_type = out_types[first_idx];

        auto inp_ptrs_slice = inp_pointers_device.get_slice(first, last);
        auto out_ptrs_slice = out_pointers_device.get_slice(first, last);
        auto inp_strs_slice = inp_strides_device.get_slice(first, last);
        auto out_strs_slice = out_strides_device.get_slice(first, last);

        const auto* const inp_ptrs = inp_ptrs_slice.get_data();
        const auto* const out_ptrs = out_ptrs_slice.get_data();
        const auto* const inp_strs = inp_strs_slice.get_data();
        const auto* const out_strs = out_strs_slice.get_data();

        const auto chunk_shape = std::make_pair(chunk_len, col_count);

        last_event = copy_convert(queue,
                                  inp_ptrs,
                                  inp_type,
                                  inp_strs,
                                  out_ptrs,
                                  out_type,
                                  out_strs,
                                  chunk_shape,
                                  { last_event });

        first = last;
    }

    return last_event;
}

} // namespace oneapi::dal::backend
