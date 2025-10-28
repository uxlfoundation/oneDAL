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

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"
#include "oneapi/dal/table/backend/convert/copy_convert.hpp"
#include "oneapi/dal/table/backend/convert/copy_convert_impl.hpp"

namespace oneapi::dal::backend {

template <typename Left, typename Right>
using longer_preferred_vector_desc_t = std::conditional_t<sizeof(Right) < sizeof(Left),
                                                          preferred_vector_desc_t<Left>,
                                                          preferred_vector_desc_t<Right>>;

template <typename InpType, typename OutType>
auto propose_range(const sycl::queue& queue, const shape_t& shape) {
    using prop_t = longer_preferred_vector_desc_t<InpType, OutType>;

    const auto row_count = shape.first;
    const auto col_count = shape.second;
    const sycl::device device = queue.get_device();
    const std::size_t vec = device.template get_info<prop_t>();
    const auto pref_vec = detail::integral_cast<std::int64_t>(vec);
    const std::int64_t count = std::max(pref_vec, col_count);
    return std::make_pair(row_count, count);
}

template <typename InpType, typename OutType>
sycl::event copy_convert_impl(sycl::queue& queue,
                              const InpType* const* inp_pointers,
                              const std::int64_t* inp_strides,
                              OutType* const* out_pointers,
                              const std::int64_t* out_strides,
                              const shape_t& shape,
                              const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto row_count = shape.first;
        const auto col_count = shape.second;
        const auto range_pair = propose_range<InpType, OutType>(queue, shape);
        const auto range_rows = range_pair.first;
        const auto range_cols = range_pair.second;
        const sycl::range<2> range_2d{ //
                                       detail::integral_cast<std::size_t>(range_rows),
                                       detail::integral_cast<std::size_t>(range_cols)
        };

        const std::int64_t wi_per_row = range_cols;

        h.parallel_for(range_2d, [=](sycl::id<2> idx) -> void {
            const std::int64_t row = idx[0];
            const std::int64_t loc = idx[1];

            const auto out_str = out_strides[row];
            const auto inp_str = inp_strides[row];

            OutType* const out_ptr = out_pointers[row];
            const InpType* const inp_ptr = inp_pointers[row];

            for (std::int64_t col = loc; col < col_count; col += wi_per_row) {
                const std::int64_t out_offset = col * out_str;
                const std::int64_t inp_offset = col * inp_str;

                OutType value = inp_ptr[inp_offset];
                out_ptr[out_offset] = std::move(value);
            }
        });
    });
}

template <typename InputType>
sycl::event copy_convert(sycl::queue& queue,
                         const InputType* const* inp_pointers,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         data_type out_type,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    ONEDAL_ASSERT(is_known_data_type(out_type));

    const auto functor = [&](auto out) -> sycl::event {
        using out_t = std::decay_t<decltype(out)>;

        auto* const conv_out_ptrs = reinterpret_cast<out_t* const*>(out_pointers);

        return copy_convert_impl<InputType, out_t>(queue,
                                                   inp_pointers,
                                                   inp_strides,
                                                   conv_out_ptrs,
                                                   out_strides,
                                                   shape,
                                                   deps);
    };

    return dispatch_by_data_type(out_type, functor);
}

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         data_type inp_type,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         data_type out_type,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    ONEDAL_ASSERT(is_known_data_type(inp_type));

    const auto functor = [&](auto inp) -> sycl::event {
        using inp_t = std::decay_t<decltype(inp)>;

        auto* const conv_inp_ptrs = reinterpret_cast<const inp_t* const*>(inp_pointers);

        return copy_convert<inp_t>(queue,
                                   conv_inp_ptrs,
                                   inp_strides,
                                   out_pointers,
                                   out_type,
                                   out_strides,
                                   shape,
                                   deps);
    };

    return dispatch_by_data_type(inp_type, functor);
}

} // namespace oneapi::dal::backend
