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
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview::triangle_counting {

namespace task {
struct local {};
struct global {};
struct local_and_global {};
using by_default = local;
} // namespace task

namespace method {
struct ordered_count {};
using by_default = ordered_count;
} // namespace method

// Kind of triangles to compute
enum class kind { undirected_clique, directed_cycle, directed_closed_triplet };

// Option to allow relabeling that is potentially additional memory consuming.
enum class relabel { no, yes };

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename T>
using enable_if_local_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::local, task::local_and_global>>;

template <typename T>
using enable_if_global_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::global, task::local_and_global>>;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::ordered_count>;

template <typename Task>
constexpr bool is_valid_task =
    dal::detail::is_one_of_v<Task, task::global, task::local_and_global, task::local>;

/// The base class for the Triangle Counting algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    kind get_kind() const;
    relabel get_relabel() const;

protected:
    void set_kind(kind value);
    void set_relabel(relabel value);

    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace detail

/// Class for the Triangle Counting algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
/// @tparam Task   The task to solve by the algorithm
/// @tparam Allocator   Custom allocator for all memory management inside the algorithm
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Allocator = std::allocator<char>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_method<Method>);
    static_assert(detail::is_valid_task<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using allocator_t = Allocator;

    explicit descriptor(Allocator allocator = std::allocator<char>()) {
        alloc_ = allocator;
    }

    kind get_kind() const {
        return base_t::get_kind();
    }

    auto& set_kind(kind value) {
        base_t::set_kind(value);
        return *this;
    }

    relabel get_relabel() const {
        return base_t::get_relabel();
    }

    auto& set_relabel(relabel value) {
        base_t::set_relabel(value);
        return *this;
    }

    Allocator get_allocator() const {
        return alloc_;
    }

private:
    Allocator alloc_;
};

namespace detail {

#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
struct is_device_csr_topology : std::false_type {};

template <typename Index, typename Weight>
struct is_device_csr_topology<dal::preview::detail::device_csr_topology<Index, Weight>>
        : std::true_type {};

template <typename Graph>
constexpr bool is_valid_graph =
    is_device_csr_topology<Graph>::value ||
    dal::detail::is_one_of_v<Graph,
                             undirected_adjacency_vector_graph<vertex_user_value_type<Graph>,
                                                               edge_user_value_type<Graph>,
                                                               graph_user_value_type<Graph>,
                                                               vertex_type<Graph>,
                                                               graph_allocator<Graph>>>;
#else
template <typename Graph>
constexpr bool is_valid_graph =
    dal::detail::is_one_of_v<Graph,
                             undirected_adjacency_vector_graph<vertex_user_value_type<Graph>,
                                                               edge_user_value_type<Graph>,
                                                               graph_user_value_type<Graph>,
                                                               vertex_type<Graph>,
                                                               graph_allocator<Graph>>>;
#endif

} // namespace detail
} // namespace oneapi::dal::preview::triangle_counting
