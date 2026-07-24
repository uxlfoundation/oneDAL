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

#pragma once

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview::shortest_paths {

namespace task {
struct one_to_all {}; // one vertex to all paths
using by_default = one_to_all;
} // namespace task

namespace method {
struct delta_stepping {};
using by_default = delta_stepping;
} // namespace method

class optional_result_id {
    using bitset_t = std::uint64_t;

public:
    optional_result_id() : mask_(0) {}

    optional_result_id(const bitset_t& mask) : mask_(mask) {}

    const bitset_t& get_mask() const {
        return mask_;
    }

    operator bool() const {
        return (mask_ > 0);
    }

    static optional_result_id get_result_id_by_index(std::int64_t result_index) {
        return optional_result_id{}.set_mask(std::uint64_t(1) << result_index);
    }

private:
    optional_result_id& set_mask(const bitset_t& mask) {
        this->mask_ = mask;
        return *this;
    }

    bitset_t mask_;
};

inline optional_result_id operator|(const optional_result_id& lhs, const optional_result_id& rhs) {
    return optional_result_id{ lhs.get_mask() | rhs.get_mask() };
}

inline optional_result_id operator&(const optional_result_id& lhs, const optional_result_id& rhs) {
    return optional_result_id{ lhs.get_mask() & rhs.get_mask() };
}

inline optional_result_id operator==(const optional_result_id& lhs, const optional_result_id& rhs) {
    return optional_result_id{ lhs.get_mask() == rhs.get_mask() };
}

inline optional_result_id operator!=(const optional_result_id& lhs, const optional_result_id& rhs) {
    return optional_result_id{ lhs.get_mask() != rhs.get_mask() };
}

namespace detail {
ONEDAL_EXPORT optional_result_id get_predecessors_id();
ONEDAL_EXPORT optional_result_id get_distances_id();
} // namespace detail

namespace optional_results {
const optional_result_id predecessors = detail::get_predecessors_id();
const optional_result_id distances = detail::get_distances_id();
} // namespace optional_results

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename T>
using enable_if_single_source_t = std::enable_if_t<dal::detail::is_one_of_v<T, task::one_to_all>>;

template <typename T, typename M>
using enable_if_delta_stepping_single_source_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::one_to_all> &
                     dal::detail::is_one_of_v<M, method::delta_stepping>>;

template <typename M>
using enable_if_delta_stepping_t =
    std::enable_if_t<dal::detail::is_one_of_v<M, method::delta_stepping>>;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::delta_stepping>;

template <typename Task>
constexpr bool is_valid_task = dal::detail::is_one_of_v<Task, task::one_to_all>;

/// The base class for the Shortest Paths algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_source() const;
    double get_delta() const;
    optional_result_id& get_optional_results() const;

protected:
    void set_source(std::int64_t source_vertex);
    void set_delta(double delta);
    void set_optional_results(const optional_result_id& optional_results);

    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace detail

/// Class for the Shortest Paths algorithm descriptor
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

    template <typename T = Task,
              typename M = Method,
              typename = detail::enable_if_delta_stepping_single_source_t<T, M>>
    explicit descriptor(std::int64_t source_vertex,
                        double delta,
                        optional_result_id optional_results = optional_results::distances,
                        const Allocator& allocator = std::allocator<char>()) {
        base_t::set_source(source_vertex);
        base_t::set_delta(delta);
        base_t::set_optional_results(optional_results);
        alloc_ = allocator;
    }

    template <typename T = Task, typename = detail::enable_if_single_source_t<T>>
    auto& set_source(std::int64_t source_vertex) {
        base_t::set_source(source_vertex);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_single_source_t<T>>
    std::int64_t get_source() const {
        return base_t::get_source();
    }

    template <typename M = Method, typename = detail::enable_if_delta_stepping_t<M>>
    auto& set_delta(double delta) {
        base_t::set_delta(delta);
        return *this;
    }

    template <typename M = Method, typename = detail::enable_if_delta_stepping_t<M>>
    double get_delta() const {
        return base_t::get_delta();
    }

    auto& set_optional_results(const optional_result_id& optional_results) {
        base_t::set_optional_results(optional_results);
        return *this;
    }

    optional_result_id& get_optional_results() const {
        return base_t::get_optional_results();
    }

    Allocator get_allocator() const {
        return alloc_;
    }

private:
    Allocator alloc_;
};

namespace detail {

template <typename Graph>
constexpr bool is_valid_graph =
    is_directed<Graph> &&
    dal::detail::is_one_of_v<edge_user_value_type<Graph>, std::int32_t, double>;

} // namespace detail
} // namespace oneapi::dal::preview::shortest_paths
