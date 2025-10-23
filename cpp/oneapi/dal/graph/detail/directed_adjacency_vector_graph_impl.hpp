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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/graph/detail/csr_topology.hpp"

namespace oneapi::dal::preview::detail {

template <typename VertexValue>
using vertex_values = container<VertexValue>;

template <typename EdgeValue>
using edge_values = container<EdgeValue>;

template <typename VertexValue = empty_value,
          typename EdgeValue = empty_value,
          typename GraphValue = empty_value,
          typename IndexType = std::int32_t,
          typename Allocator = std::allocator<char>>
class ONEDAL_EXPORT directed_adjacency_vector_graph_impl {
public:
    using allocator_type = Allocator;

    using topology_type = topology<IndexType>;

    // graph weight types
    using graph_user_value_type = GraphValue;
    using const_graph_user_value_type = const graph_user_value_type;

    using vertex_type = typename topology_type::vertex_type;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;

    using vertex_set = typename topology_type::vertex_set;
    using vertex_iterator = typename topology_type::vertex_iterator;
    using const_vertex_iterator = typename topology_type::const_vertex_iterator;
    using vertex_size_type = typename topology_type::vertex_size_type;

    using vertex_edge_type = typename topology_type::vertex_edge_type;
    using vertex_edge_size_type = typename topology_type::vertex_edge_size_type;
    using vertex_edge_set = typename topology_type::vertex_edge_set;
    using vertex_edge_iterator = typename topology_type::vertex_edge_iterator;
    using const_vertex_edge_iterator = typename topology_type::const_vertex_edge_iterator;
    using vertex_edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_edge_type>;

    // vertex edge types
    using vertex_outward_edge_type = typename topology_type::vertex_edge_type;
    using vertex_outward_edge_size_type = typename topology_type::vertex_edge_size_type;
    using vertex_outward_edge_set = typename topology_type::vertex_edge_set;
    using vertex_outward_edge_iterator = typename topology_type::vertex_edge_iterator;
    using const_vertex_outward_edge_iterator = typename topology_type::const_vertex_edge_iterator;
    using vertex_outward_edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_edge_type>;

    using edge_type = typename topology_type::edge_type;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_set = typename topology_type::edge_set;

    using edge_size_type = typename topology_type::edge_size_type;

    using vertex_user_value_type = VertexValue;
    using vertex_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_user_value_type>;
    using vertex_user_value_set = container<vertex_user_value_type>;

    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_set = container<edge_user_value_type>;
    using const_edge_value_iterator = const edge_user_value_type*;

    // ranges
    using vertex_edge_range = typename topology_type::vertex_edge_range;
    using const_vertex_edge_range = typename topology_type::const_vertex_edge_range;

    // ranges
    using vertex_outward_edge_range = typename topology_type::vertex_edge_range;
    using const_vertex_outward_edge_range = typename topology_type::const_vertex_edge_range;
    using const_edge_value_range_type = range<const_edge_value_iterator>;

    directed_adjacency_vector_graph_impl() = default;

    ~directed_adjacency_vector_graph_impl() = default;

    template <typename... Args>
    inline void set_topology(Args&&... args) {
        _topology.set_topology(std::forward<Args>(args)...);
    }

    inline void set_edge_values(EdgeValue* values, std::int64_t values_count) {
        _edge_values = edge_values<EdgeValue>::wrap(values, values_count);
    }

    inline void set_edge_values(const EdgeValue* values, std::int64_t values_count) {
        _edge_values = edge_values<EdgeValue>::wrap(values, values_count);
    }

    inline void set_edge_values(edge_values<EdgeValue>& edge_values_array) {
        _edge_values = edge_values_array;
    }

    inline topology<IndexType>& get_topology() {
        return _topology;
    }

    inline vertex_values<VertexValue>& get_vertex_values() {
        return _vertex_values;
    }

    inline edge_values<EdgeValue>& get_edge_values() {
        return _edge_values;
    }

    inline const topology<IndexType> get_topology() const {
        return _topology;
    }

    inline const vertex_values<VertexValue> get_vertex_values() const {
        return _vertex_values;
    }

    inline const edge_values<EdgeValue> get_edge_values() const {
        return _edge_values;
    }

    inline const EdgeValue& get_edge_value(vertex_type u, vertex_type v) const {
        if (_edge_values.get_count() == 0)
            throw range_error(dal::detail::error_messages::edge_values_are_empty());
        const auto [begin, end] = _topology.get_vertex_neighbors(u);
        for (auto i = begin; i < end; i++) {
            if (v == *i) {
                return _edge_values[i - _topology._cols_ptr];
            }
        }
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }

    allocator_type _allocator;
    vertex_allocator_type _vertex_allocator{ _allocator };
    edge_allocator_type _edge_allocator{ _allocator };
    vertex_edge_allocator_type _vertex_edge_allocator{ _allocator };
    vertex_user_value_allocator_type _vertex_user_value_allocator{ _allocator };
    edge_user_value_allocator_type _edge_user_value_allocator{ _allocator };

private:
    topology<IndexType> _topology;
    vertex_values<VertexValue> _vertex_values;
    edge_values<EdgeValue> _edge_values;
};

} // namespace oneapi::dal::preview::detail
