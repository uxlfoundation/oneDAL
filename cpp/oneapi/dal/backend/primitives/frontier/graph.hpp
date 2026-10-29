/*******************************************************************************
* Copyright contributors to the oneDAL project
* Copyright 2025 University of Salerno
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
#include "oneapi/dal/graph/detail/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

template <typename VertexT = std::uint32_t,
          typename EdgeT = std::uint32_t,
          typename WeightT = std::uint32_t>
class csr_graph_view {
    using vertex_t = VertexT;
    using edge_t = EdgeT;
    using weight_t = WeightT;

    struct neighbor_iterator_t {
        neighbor_iterator_t(const vertex_t* start_ptr, vertex_t* ptr)
                : _ptr(ptr),
                  _start_ptr(start_ptr) {}

        SYCL_EXTERNAL bool operator==(const neighbor_iterator_t& other) const {
            return _ptr == other._ptr;
        }

        SYCL_EXTERNAL bool operator!=(const neighbor_iterator_t& other) const {
            return _ptr != other._ptr;
        }

        SYCL_EXTERNAL neighbor_iterator_t& operator++() {
            ++_ptr;
            return *this;
        }

        SYCL_EXTERNAL neighbor_iterator_t operator+(int n) const {
            neighbor_iterator_t tmp = *this;
            tmp._ptr += n;
            return tmp;
        }

        SYCL_EXTERNAL vertex_t operator*() const {
            return *_ptr;
        }

        SYCL_EXTERNAL edge_t get_index() const {
            return static_cast<edge_t>(_ptr - _start_ptr);
        }

        vertex_t* _ptr;
        const vertex_t* _start_ptr;
    };

public:
    csr_graph_view(std::uint64_t num_nodes, vertex_t* row_ptr, edge_t* col_indices, weight_t* weights)
            : _num_nodes(num_nodes),
              _row_ptr(row_ptr),
              _col_indices(col_indices),
              _weights(weights) {
        ONEDAL_ASSERT(_row_ptr != nullptr, "Row pointer must not be null");
        ONEDAL_ASSERT(_col_indices != nullptr, "Column indices must not be null");
        ONEDAL_ASSERT(_weights != nullptr, "Weights must not be null");
    }

    SYCL_EXTERNAL inline std::uint32_t get_degree(const vertex_t vertex) const {
        ONEDAL_ASSERT(vertex < _num_nodes, "Vertex index out of bounds");
        return static_cast<std::uint32_t>(_row_ptr[vertex + 1] - _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline weight_t get_weight(const edge_t edge) const {
        return _weights[edge];
    }

    SYCL_EXTERNAL inline neighbor_iterator_t begin(vertex_t vertex) const {
        ONEDAL_ASSERT(vertex < _num_nodes, "Vertex index out of bounds");
        return neighbor_iterator_t(_col_indices, _col_indices + _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline neighbor_iterator_t end(vertex_t vertex) const {
        ONEDAL_ASSERT(vertex < _num_nodes, "Vertex index out of bounds");
        return neighbor_iterator_t(_col_indices, _col_indices + _row_ptr[vertex + 1]);
    }

private:
    std::uint64_t _num_nodes;
    vertex_t* _row_ptr;
    edge_t* _col_indices;
    weight_t* _weights;
};

template <typename VertexT = std::uint32_t,
          typename EdgeT = std::uint32_t,
          typename WeightT = std::uint32_t>
class csr_graph {
    using vertex_t = VertexT;
    using edge_t = EdgeT;
    using weight_t = WeightT;
    using graph_view_t = csr_graph_view<vertex_t, edge_t, weight_t>;

public:
    csr_graph(sycl::queue& queue,
              std::vector<VertexT> row_ptr,
              std::vector<EdgeT> col_indices,
              std::vector<WeightT> weights = {},
              sycl::usm::alloc alloc = sycl::usm::alloc::shared)
            : _queue(queue),
              _num_nodes(row_ptr.size() - 1) {
        std::int64_t row_ptr_size = static_cast<std::int64_t>(row_ptr.size());
        std::int64_t col_indices_size = static_cast<std::int64_t>(col_indices.size());
        std::int64_t weights_size = static_cast<std::int64_t>(weights.size());

        _row_ptr = ndarray<vertex_t, 1>::empty(_queue, { row_ptr_size }, alloc);
        _col_indices = ndarray<edge_t, 1>::empty(_queue, { col_indices_size }, alloc);
        _weights = ndarray<weight_t, 1>::empty(_queue, { weights_size }, alloc);

        auto e1 = dal::backend::copy_host2usm(_queue,
                                              _row_ptr.get_mutable_data(),
                                              row_ptr.data(),
                                              row_ptr.size());

        auto e2 = dal::backend::copy_host2usm(_queue,
                                              _col_indices.get_mutable_data(),
                                              col_indices.data(),
                                              col_indices.size());

        auto e3 = dal::backend::copy_host2usm(_queue,
                                              _weights.get_mutable_data(),
                                              weights.data(),
                                              weights.size());

        e1.wait_and_throw();
        e2.wait_and_throw();
        e3.wait_and_throw();
    }

    graph_view_t get_device_view() const {
        return { _num_nodes,
                 _row_ptr.get_mutable_data(),
                 _col_indices.get_mutable_data(),
                 _weights.get_mutable_data() };
    }

    sycl::queue& get_queue() const {
        return _queue;
    }

    std::uint64_t get_vertex_count() const {
        return _num_nodes;
    }

private:
    sycl::queue& _queue;
    std::uint64_t _num_nodes;
    ndarray<vertex_t, 1> _row_ptr;
    ndarray<edge_t, 1> _col_indices;
    ndarray<weight_t, 1> _weights;
};

} // namespace oneapi::dal::backend::primitives
