/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview::jaccard {

namespace task {
struct all_vertex_pairs {};
using by_default = all_vertex_pairs;
} // namespace task

namespace method {
struct fast {};
using by_default = fast;
} // namespace method

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::fast>;

template <typename Task>
constexpr bool is_valid_task = dal::detail::is_one_of_v<Task, task::all_vertex_pairs>;

/// The base class for the Jaccard similarity algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    auto get_row_range_begin() const -> std::int64_t;
    auto get_row_range_end() const -> std::int64_t;
    auto get_column_range_begin() const -> std::int64_t;
    auto get_column_range_end() const -> std::int64_t;

protected:
    void set_row_range_impl(std::int64_t begin, std::int64_t end);
    void set_column_range_impl(std::int64_t begin, std::int64_t end);
    void set_block_impl(const std::initializer_list<std::int64_t>& row_range,
                        const std::initializer_list<std::int64_t>& column_range);

    dal::detail::pimpl<detail::descriptor_impl<task_t>> impl_;
};

} // namespace detail

/// Class for the Jaccard similarity algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_method<Method>);
    static_assert(detail::is_valid_task<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    descriptor() = default;

    /// Returns the beginning of the row of the graph block
    std::int64_t get_row_range_begin() const {
        return base_t::get_row_range_begin();
    }

    /// Returns the end of the row of the graph block
    std::int64_t get_row_range_end() const {
        return base_t::get_row_range_end();
    }

    /// Returns the beginning of the column of the graph block
    std::int64_t get_column_range_begin() const {
        return base_t::get_column_range_begin();
    }

    /// Returns the end of the column of the graph block
    std::int64_t get_column_range_end() const {
        return base_t::get_column_range_end();
    }

    /// Sets the range of the rows of the graph block for Jaccard similarity computation
    ///
    /// @param [in] begin  The begin of the row of the graph block
    /// @param [in] end    The end of the row of the graph block
    auto& set_row_range(std::int64_t begin, std::int64_t end) {
        base_t::set_row_range_impl(begin, end);
        return *this;
    }

    /// Sets the range of the columns of the graph block for Jaccard similarity computation
    ///
    /// @param [in] begin  The begin of the column of the graph block
    /// @param [in] end    The end of the column of the graph block
    auto& set_column_range(std::int64_t begin, std::int64_t end) {
        base_t::set_column_range_impl(begin, end);
        return *this;
    }

    /// Sets the range of the rows and columns of the graph block for Jaccard similarity
    /// computation
    ///
    /// @param [in] row_range     The range of the rows of the graph block
    /// @param [in] column_range  The range of the columns of the graph block
    auto& set_block(const std::initializer_list<std::int64_t>& row_range,
                    const std::initializer_list<std::int64_t>& column_range) {
        base_t::set_block_impl(row_range, column_range);
        return *this;
    }
};

/// Structure for the caching builder
struct ONEDAL_EXPORT caching_builder {
    /// Returns the pointer to the allocated memory of size block_max_size.
    ///
    /// @param [in]   block_max_size  The required size of memory
    void* operator()(std::int64_t block_max_size);

    std::shared_ptr<byte_t> result_ptr;
    std::int64_t size = 0;
};

namespace detail {

template <typename Graph>
constexpr bool is_valid_graph =
    dal::detail::is_one_of_v<Graph,
                             undirected_adjacency_vector_graph<vertex_user_value_type<Graph>,
                                                               edge_user_value_type<Graph>,
                                                               graph_user_value_type<Graph>,
                                                               vertex_type<Graph>,
                                                               graph_allocator<Graph>>>;

} // namespace detail

} // namespace oneapi::dal::preview::jaccard
