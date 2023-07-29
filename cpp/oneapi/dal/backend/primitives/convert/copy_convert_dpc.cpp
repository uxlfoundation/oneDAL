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

#pragma once

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/atomic.hpp"

#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

template <typename OutputType>
struct naive_kernel_state {

};

template <typename InputType, typename OutputType, typename Index>
inline void naive_kernel_impl(const naive_kernel_state<OutputType>& k, const Index& idx) {

}

template <typename OutputType>
class naive_kernel : public naive_kernel_state<OutputType> {
    using base_t = naive_kernel_state<OutputType>;
public:
    naive_copy_convert_kernel() = delete;

    naive_copy_convert_kernel()

    naive_copy_convert_kernel(naive_copy_convert_kernel&&) = default;
    naive_copy_convert_kernel(const naive_copy_convert_kernel&) = default;

    void operator() (sycl::id<2> idx) const {
        const auto body = [&](auto type) -> void {
            using input_type_t = std::remove_cv_t<decltype(type)>;
            naive_kernel_impl<input_type_t>(*this, idx);
        };

        const auto dtype = *(type_ptr + idx.get(0));
        const auto on_unknown = [](data_type) -> void {};
        detail::dispatch_by_dtype(dtype, body, on_unknown);
    }

private:
    const std::size_t str;
    const std::size_t width;
    OutputType* const out_ptr;
    const std::size_t row_stride;
    const std::size_t col_stride;
    const dal::byte* const inp_ptr;
    const data_type* const type_ptr;
    const std::size_t* const off_ptr;
};

auto get_range(const sycl::queue& q, const shape_t& s) {
    const auto r_count = detail::integral_cast<std::size_t>(s.first);
    const auto c_count = detail::integral_cast<std::size_t>(s.second);

    return std::make_pair(r_count, c_count);
}

sycl::event copy_convert(sycl::queue& queue,
                         const dal::array<data_type>& input_types,
                         const dal::array<dal::byte>& input_data,
                         const dal::array<std::size_t>& input_offsets,
                         const shape_t& input_shape,
                         data_type output_type,
                         dal::array<dal::byte>& output_data,
                         const shape_t& output_strides,
                         const std::vector<sycl::event>& deps = {}) {
    check_dimensions(input_types, input_data, input_shape,
                     output_type, output_data, output_strides);

    const auto naive_impl = [&](auto type) -> sycl::event {
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);

            const auto raw_range = get_range(queue, input_shape);

            const sycl::range<2> range{ raw_range.first, raw_range.second };

            const auto* const inp_offsets_ptr = input_offsets.get_data();
            const auto* const inp_types_ptr = input_types.get_data();
            const auto* const inp_data_ptr = input_data.get_data();

            using output_type_t = std::remove_cv<decltype(type)>;
            auto* const raw_out_ptr = output_data.get_mutable_data();
            auto* const out_data_ptr = reinterpret_cast<output_type_t*>();

            const naive_copy_convert_kernel<output_type_t> kernel{

            };

            h.parallel_for(range, kernel);
        });
    }

    return detail::dispatch_by_data_type(output_type, naive_impl);
}

} // namespace oneapi::dal::backend::primitives