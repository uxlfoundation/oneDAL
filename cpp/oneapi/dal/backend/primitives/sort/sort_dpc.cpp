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

#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-local-typedef"

#include <oneapi/dpl/experimental/kernel_templates>
#pragma clang diagnostic pop

namespace oneapi::dal::backend::primitives {

namespace de = dal::detail;

using sycl::ext::oneapi::plus;

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace(sycl::queue& queue,
                                       ndview<Float, 1>& val_in,
                                       ndview<Index, 1>& ind_in,
                                       const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort_indices_inplace, queue);
    ONEDAL_ASSERT(val_in.has_mutable_data());
    ONEDAL_ASSERT(ind_in.has_mutable_data());
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());

    if (val_in.get_count() > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_elements_to_sort());
    }

    auto event = oneapi::dpl::experimental::kt::gpu::esimd::radix_sort_by_key<true, 8>(
        queue,
        val_in.get_mutable_data(),
        val_in.get_mutable_data() + val_in.get_count(),
        ind_in.get_mutable_data(),
        dpl::experimental::kt::kernel_param<256, 32>{});
    return event;
}

template <typename Integer>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<Integer, 2>& val_in,
                       ndview<Integer, 2>& val_out,
                       std::int64_t sorted_elem_count,
                       const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue);

    const auto row_count = val_in.get_dimension(0);
    const auto col_count = val_in.get_dimension(1);
    sycl::event radix_sort_event;

    for (std::int64_t row = 0; row < row_count; ++row) {
        Integer* row_start_in = val_in.get_mutable_data() + row * col_count;
        Integer* row_start_out = val_out.get_mutable_data() + row * col_count;

        const auto row_sorted_elem_count = std::min(sorted_elem_count, col_count);

        radix_sort_event = oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<true, 8>(
            queue,
            row_start_in,
            row_start_in + row_sorted_elem_count,
            row_start_out,
            dpl::experimental::kt::kernel_param<256, 32>{});
    }

    return radix_sort_event;
}

template <typename Integer>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<Integer, 2>& val_in,
                       ndview<Integer, 2>& val_out,
                       const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue);
    return radix_sort(queue, val_in, val_out, val_in.get_dimension(1), deps);
}

#define INSTANTIATE_SORT_INDICES(Float, Index)                                   \
    template ONEDAL_EXPORT sycl::event radix_sort_indices_inplace<Float, Index>( \
        sycl::queue&,                                                            \
        ndview<Float, 1>&,                                                       \
        ndview<Index, 1>&,                                                       \
        const event_vector&);

#define INSTANTIATE_FLOAT(Index)           \
    INSTANTIATE_SORT_INDICES(float, Index) \
    INSTANTIATE_SORT_INDICES(double, Index)

INSTANTIATE_FLOAT(std::uint32_t)
INSTANTIATE_FLOAT(std::int32_t)

#define INSTANTIATE_RADIX_SORT(Index)                                       \
    template ONEDAL_EXPORT sycl::event radix_sort<Index>(sycl::queue&,      \
                                                         ndview<Index, 2>&, \
                                                         ndview<Index, 2>&, \
                                                         const event_vector&);

INSTANTIATE_RADIX_SORT(std::int32_t)
INSTANTIATE_RADIX_SORT(std::uint32_t)
INSTANTIATE_RADIX_SORT(std::int64_t)
INSTANTIATE_RADIX_SORT(std::uint64_t)

#define INSTANTIATE_RADIX_SORT_WITH_COUNT(Index)                            \
    template ONEDAL_EXPORT sycl::event radix_sort<Index>(sycl::queue&,      \
                                                         ndview<Index, 2>&, \
                                                         ndview<Index, 2>&, \
                                                         std::int64_t,      \
                                                         const event_vector&);

INSTANTIATE_RADIX_SORT_WITH_COUNT(std::int32_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::uint32_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::int64_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::uint64_t)

} // namespace oneapi::dal::backend::primitives
