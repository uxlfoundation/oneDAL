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

#include <deque>
#include <limits>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

template <typename T,
          ndorder order = ndorder::c,
          typename Container = std::deque<ndarray<T, 2, order>>>
inline auto& split_table_inplace(const table& input,
                                 std::int64_t block,
                                 Container& container,
                                 T default_value = std::numeric_limits<T>::max()) {
    static_assert(std::is_same_v<typename Container::value_type, ndarray<T, 2, order>>);

    ONEDAL_ASSERT(input.has_data());
    row_accessor<const T> accessor{ input };
    const auto row_count = input.get_row_count();
    const auto col_count = input.get_column_count();

    uniform_blocking blocking{ row_count, block };
    const auto blk_count = blocking.get_block_count();
    for (std::int64_t b = 0; b < blk_count; ++b) {
        const auto f_row = blocking.get_block_start_index(b);
        const auto l_row = blocking.get_block_end_index(b);
        const auto len = l_row - f_row;

        const auto raw_array = accessor.pull({ f_row, l_row });
        const auto raw_view = ndview<T, 2>::wrap(raw_array.get_data(), { len, col_count });

        auto tmp = ndarray<T, 2, order>::empty({ block, col_count });
        auto tmp_slice = tmp.get_row_slice(0, len);
        if (len != block)
            tmp.fill(default_value);

        copy(tmp_slice, raw_view);

        container.push_back(std::move(tmp));
    }

    return container;
}

template <typename T,
          ndorder order = ndorder::c,
          typename Container = std::deque<ndarray<T, 2, order>>>
inline auto split_table(const table& input,
                        std::int64_t block,
                        T default_value = std::numeric_limits<T>::max()) {
    Container result;
    split_table_inplace<T, order>(input, block, result, default_value);
    return result;
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename T,
          ndorder order = ndorder::c,
          typename Container = std::deque<ndarray<T, 2, order>>>
inline auto& split_table_inplace(sycl::queue& queue,
                                 const table& input,
                                 std::int64_t block,
                                 Container& container,
                                 T default_value = std::numeric_limits<T>::max(),
                                 sycl::usm::alloc kind = sycl::usm::alloc::device) {
    static_assert(std::is_same_v<typename Container::value_type, ndarray<T, 2, order>>);

    ONEDAL_ASSERT(input.has_data());
    row_accessor<const T> accessor{ input };
    const auto row_count = input.get_row_count();
    const auto col_count = input.get_column_count();

    uniform_blocking blocking{ row_count, block };
    const auto blk_count = blocking.get_block_count();

    for (std::int64_t b = 0; b < blk_count; ++b) {
        const auto f_row = blocking.get_block_start_index(b);
        const auto l_row = blocking.get_block_end_index(b);
        const auto len = l_row - f_row;
        const auto raw_array = accessor.pull(queue, { f_row, l_row }, kind);
        const auto raw_view = ndview<T, 2>::wrap(raw_array.get_data(), { len, col_count });

        auto tmp = ndarray<T, 2, order>::empty(queue, { block, col_count }, kind);
        auto tmp_slice = tmp.get_row_slice(0, len);

        auto fevent = len != block ? fill(queue, tmp, default_value) : sycl::event{};
        
        // Small hotfix to call wait_and_throw inside the loop
        // Investigation issue with creating an event_vector and calling wait_and_throw outside loop is needed
        copy(queue, tmp_slice, raw_view, { fevent }).wait_and_throw();
        container.push_back(std::move(tmp));
    }

    return container;
}

template <typename T,
          ndorder order = ndorder::c,
          typename Container = std::deque<ndarray<T, 2, order>>>
inline auto split_table(sycl::queue& queue,
                        const table& input,
                        std::int64_t block,
                        T default_value = std::numeric_limits<T>::max(),
                        sycl::usm::alloc kind = sycl::usm::alloc::device) {
    Container result;
    {
        ONEDAL_PROFILER_TASK(split_table, queue);
        split_table_inplace<T, order>(queue, input, block, result, default_value, kind);
    }
    return result;
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
