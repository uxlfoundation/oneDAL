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
class ONEDAL_EXPORT undirected_adjacency_vector_graph_impl {
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

    // ranges
    using vertex_edge_range = typename topology_type::vertex_edge_range;
    using const_vertex_edge_range = typename topology_type::const_vertex_edge_range;

    undirected_adjacency_vector_graph_impl() = default;

    ~undirected_adjacency_vector_graph_impl() = default;

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

#ifdef ONEDAL_DATA_PARALLEL
    inline void to_device(sycl::queue& queue) {
        const auto* ptr = _topology._rows.get_data();
        if (ptr != nullptr) {
            auto alloc_type = sycl::get_pointer_type(ptr, queue.get_context());
            if (alloc_type == sycl::usm::alloc::unknown || alloc_type == sycl::usm::alloc::host) {
                _topology = _topology.to_device(queue);
            }
        }

        if constexpr (!std::is_same_v<VertexValue, empty_value>) {
            transfer_array_to_device(queue, _vertex_values);
        }

        if constexpr (!std::is_same_v<EdgeValue, empty_value>) {
            transfer_array_to_device(queue, _edge_values);
        }
    }

private:
    template <typename T>
    static void transfer_array_to_device(sycl::queue& queue, container<T>& arr) {
        const auto count = arr.get_count();
        if (count == 0)
            return;
        const auto* src = arr.get_data();
        auto alloc_type = sycl::get_pointer_type(src, queue.get_context());
        if (alloc_type == sycl::usm::alloc::device || alloc_type == sycl::usm::alloc::shared) {
            return;
        }
        auto device_arr = container<T>::empty(queue, count, sycl::usm::alloc::shared);
        queue.memcpy(device_arr.get_mutable_data(), src, count * sizeof(T)).wait_and_throw();
        arr = device_arr;
    }

public:
#endif

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
