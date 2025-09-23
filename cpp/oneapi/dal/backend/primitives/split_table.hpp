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
    auto all_rows = accessor.pull({ 0, row_count });
    std::cout << "Input table (first 3 rows):" << std::endl;
    for (std::int64_t row = 0; row < std::min<std::int64_t>(3, row_count); ++row) {
        for (std::int64_t col = 0; col < std::min<std::int64_t>(3, col_count); ++col) {
            std::cout << all_rows[row * col_count + col] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "BLOCK COUNT and size and default: " << blk_count << " " << block << " " << default_value << std::endl;
    for (std::int64_t b = 0; b < blk_count; ++b) {
        const auto f_row = blocking.get_block_start_index(b);
        const auto l_row = blocking.get_block_end_index(b);
        const auto len = l_row - f_row;
        std::cout << "BLOCK INDEX: " << b <<  " start and end incides: " << f_row << " " << l_row << std::endl;

        const auto raw_array = accessor.pull(queue, { f_row, l_row }, kind);
        const auto raw_view = ndview<T, 2>::wrap(raw_array.get_data(), { len, col_count });
        auto host_raw_view = raw_view.to_host(queue);
        std::cout << "raw_view (host) for block " << b << " (first 3 rows):" << std::endl;
        for (std::int64_t row = 0; row < std::min<std::int64_t>(3, host_raw_view.get_dimension(0)); ++row) {
            for (std::int64_t col = 0; col < std::min<std::int64_t>(3, host_raw_view.get_dimension(1)); ++col) {
                std::cout << host_raw_view.at(row, col) << " ";
            }
            std::cout << std::endl;
        }
        auto tmp = ndarray<T, 2, order>::empty(queue, { block, col_count }, kind);
        auto tmp_slice = tmp.get_row_slice(0, len);

        auto fevent = len != block ? fill(queue, tmp, default_value) : sycl::event{};
        
        // Small hotfix to call wait_and_throw inside the loop
        // Investigation issue with creating an event_vector and calling wait_and_throw outside loop is needed
        copy(queue, tmp_slice, raw_view, { fevent }).wait_and_throw();
        container.push_back(std::move(tmp));
    }

    // Debug print: show first few rows/cols of each block
    for (std::size_t b = 0; b < container.size(); ++b) {
        auto host_tmp = container[b].to_host(queue);
        std::cout << "split_table SYCL block " << b << " (first 3 rows):" << std::endl;
        for (std::int64_t row = 0; row < std::min<std::int64_t>(3, host_tmp.get_dimension(0)); ++row) {
            for (std::int64_t col = 0; col < std::min<std::int64_t>(3, host_tmp.get_dimension(1)); ++col) {
                std::cout << host_tmp.at(row, col) << " ";
            }
            std::cout << std::endl;
        }
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
