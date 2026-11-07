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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

namespace oneapi::dal::preview::detail {

template <typename IndexType>
class topology {
public:
    using vertex_type = IndexType;
    using vertex_set = container<vertex_type>;
    using vertex_iterator = vertex_type*;
    using const_vertex_iterator = const vertex_type*;
    using vertex_size_type = std::int64_t;
    using vertex_edge_type = vertex_type;
    using vertex_edge_size_type = vertex_type;

    using edge_type = std::int64_t;
    using edge_set = container<edge_type>;

    using vertex_edge_set = container<vertex_edge_type>;
    using vertex_edge_iterator = vertex_edge_type*;
    using const_vertex_edge_iterator = const vertex_edge_type*;
    using edge_size_type = std::int64_t;

    // ranges
    using vertex_edge_range = range<vertex_edge_iterator>;
    using const_vertex_edge_range = range<const_vertex_edge_iterator>;

    topology() = default;
    ~topology() = default;

    inline void set_topology(vertex_set& cols,
                             edge_set& rows,
                             vertex_set& degrees,
                             edge_size_type edge_count) {
        _vertex_count = degrees.get_count();
        _edge_count = edge_count;
        _cols = cols;
        _rows = rows;
        _degrees = degrees;
        _cols_ptr = _cols.get_mutable_data();
        _rows_ptr = _rows.get_mutable_data();
        _degrees_ptr = _degrees.get_mutable_data();
    }

    inline void set_topology(vertex_size_type vertex_count,
                             edge_size_type edge_count,
                             edge_type* offsets,
                             vertex_type* neighbors,
                             edge_size_type neighbors_count,
                             vertex_type* degrees) {
        _vertex_count = vertex_count;
        _edge_count = edge_count;
        _rows = edge_set::wrap(offsets, vertex_count + 1);
        _degrees = vertex_set::wrap(degrees, vertex_count);
        _cols = vertex_set::wrap(neighbors, neighbors_count);
        _rows_ptr = _rows.get_data();
        _cols_ptr = _cols.get_data();
        _degrees_ptr = _degrees.get_data();
    }

    inline void set_topology(vertex_size_type vertex_count,
                             edge_size_type edge_count,
                             const edge_type* offsets,
                             const vertex_type* neighbors,
                             edge_size_type neighbors_count,
                             const vertex_type* degrees) {
        _vertex_count = vertex_count;
        _edge_count = edge_count;
        _rows = edge_set::wrap(offsets, vertex_count + 1);
        _degrees = vertex_set::wrap(degrees, vertex_count);
        _cols = vertex_set::wrap(neighbors, neighbors_count);
        _rows_ptr = _rows.get_data();
        _cols_ptr = _cols.get_data();
        _degrees_ptr = _degrees.get_data();
    }

    ONEDAL_FORCEINLINE std::int64_t get_vertex_count() const {
        return _vertex_count;
    }

    ONEDAL_FORCEINLINE std::int64_t get_edge_count() const {
        return _edge_count;
    }

    ONEDAL_FORCEINLINE auto get_vertex_degree(const IndexType& vertex) const noexcept
        -> vertex_edge_size_type {
        return _degrees_ptr[vertex];
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors(const IndexType& vertex) const noexcept
        -> const_vertex_edge_range {
        const IndexType* vertex_neighbors_begin = _cols.get_data() + _rows[vertex];
        const IndexType* vertex_neighbors_end = _cols.get_data() + _rows[vertex + 1];
        return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors_begin(const IndexType& vertex) const noexcept
        -> const_vertex_edge_iterator {
        return _cols_ptr + _rows[vertex];
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors_end(const IndexType& vertex) const noexcept
        -> const_vertex_edge_iterator {
        return _cols_ptr + _rows[vertex + 1];
    }

#ifdef ONEDAL_DATA_PARALLEL
    inline void set_topology(const sycl::queue& queue,
                             vertex_size_type vertex_count,
                             edge_size_type edge_count,
                             const edge_type* device_offsets,
                             const vertex_type* device_neighbors,
                             edge_size_type neighbors_count) {
        _vertex_count = vertex_count;
        _edge_count = edge_count;
        _rows = edge_set::wrap(queue, device_offsets, vertex_count + 1);
        _cols = vertex_set::wrap(queue, device_neighbors, neighbors_count);
        _rows_ptr = _rows.get_data();
        _cols_ptr = _cols.get_data();
        _degrees_ptr = nullptr;
    }

    topology<IndexType> to_device(sycl::queue& queue) const {
        topology<IndexType> result;
        result._vertex_count = _vertex_count;
        result._edge_count = _edge_count;

        const auto rows_count = _vertex_count + 1;
        const auto cols_count = _cols.get_count();
        const auto degrees_count = _degrees.get_count();
        const auto rows_vertex_count = _rows_vertex.get_count();

        auto device_rows = edge_set::empty(queue, rows_count, sycl::usm::alloc::shared);
        auto device_cols = vertex_set::empty(queue, cols_count, sycl::usm::alloc::shared);

        auto e1 = queue.memcpy(device_rows.get_mutable_data(),
                               _rows.get_data(),
                               rows_count * sizeof(edge_type));
        auto e2 = queue.memcpy(device_cols.get_mutable_data(),
                               _cols.get_data(),
                               cols_count * sizeof(vertex_type));
        e1.wait_and_throw();
        e2.wait_and_throw();

        result._rows = device_rows;
        result._cols = device_cols;
        result._rows_ptr = result._rows.get_data();
        result._cols_ptr = result._cols.get_data();

        if (degrees_count > 0) {
            auto device_degrees = vertex_set::empty(queue, degrees_count, sycl::usm::alloc::shared);
            queue
                .memcpy(device_degrees.get_mutable_data(),
                        _degrees.get_data(),
                        degrees_count * sizeof(vertex_type))
                .wait_and_throw();
            result._degrees = device_degrees;
            result._degrees_ptr = result._degrees.get_data();
        }

        if (rows_vertex_count > 0) {
            auto device_rows_vertex =
                vertex_edge_set::empty(queue, rows_vertex_count, sycl::usm::alloc::shared);
            queue
                .memcpy(device_rows_vertex.get_mutable_data(),
                        _rows_vertex.get_data(),
                        rows_vertex_count * sizeof(vertex_edge_type))
                .wait_and_throw();
            result._rows_vertex = device_rows_vertex;
        }

        return result;
    }
#endif

    vertex_set _cols;
    vertex_set _degrees;
    edge_set _rows;
    vertex_edge_set _rows_vertex;

    const vertex_type* _cols_ptr = nullptr;
    const vertex_type* _degrees_ptr = nullptr;
    const edge_type* _rows_ptr = nullptr;

    std::int64_t _vertex_count = 0;
    std::int64_t _edge_count = 0;
};

} // namespace oneapi::dal::preview::detail
