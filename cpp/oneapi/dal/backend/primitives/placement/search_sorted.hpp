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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

enum search_alignment : std::int64_t { left = 0b0l, right = 0b1l };

template <typename Type, typename Index, bool clip = false>
inline sycl::event search_sorted_1d(sycl::queue& queue,
                                    search_alignment alignment,
                                    const ndview<Type, 1>& data,
                                    const ndview<Type, 1>& points,
                                    ndview<Index, 1>& results,
                                    const event_vector& deps = {}) {
    constexpr auto left = search_alignment::left;
    constexpr auto right = search_alignment::right;

    if (alignment == left) {
        return search_sorted_1d<search_alignment::left, Type, Index, clip>(queue, data, points, results, deps);
    }

    if (alignment == right) {
        return search_sorted_1d<search_alignment::right, Type, Index, clip>(queue, data, points, results, deps);
    }

    ONEDAL_ASSERT(false);
    return wait_or_pass(deps);
}

template <search_alignment alignment, typename Type, typename Index, bool clip = false>
sycl::event search_sorted_1d(sycl::queue& queue,
                             const ndview<Type, 1>& data,
                             const ndview<Type, 1>& points,
                             ndview<Index, 1>& results,
                             const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
