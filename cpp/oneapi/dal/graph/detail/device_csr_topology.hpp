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

#ifdef ONEDAL_DATA_PARALLEL

#include <sycl/sycl.hpp>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/graph/detail/csr_topology.hpp"

namespace oneapi::dal::preview::detail {

/// A device-resident CSR topology that holds row offsets, column indices,
/// and optionally edge weights as dal::array objects in device USM memory.
///
/// Row offsets are stored as int64 (matching the host topology) and column
/// indices as IndexType (typically int32). Edge weights are optional —
/// set WeightType to void (default) for unweighted graphs.
///
/// Construct via \ref topology_to_device() or \ref topology_to_host().
template <typename IndexType = std::int32_t, typename WeightType = void>
class device_csr_topology {
public:
    using index_type = IndexType;
    using weight_type = WeightType;

    device_csr_topology() = default;

    /// Construct an unweighted device topology.
    device_csr_topology(dal::array<std::int64_t> device_rows,
                        dal::array<IndexType> device_cols,
                        std::int64_t vertex_count,
                        std::int64_t edge_count)
            : rows_(std::move(device_rows)),
              cols_(std::move(device_cols)),
              vertex_count_(vertex_count),
              edge_count_(edge_count) {}

    /// Construct a weighted device topology.
    device_csr_topology(dal::array<std::int64_t> device_rows,
                        dal::array<IndexType> device_cols,
                        dal::array<WeightType> device_weights,
                        std::int64_t vertex_count,
                        std::int64_t edge_count)
            : rows_(std::move(device_rows)),
              cols_(std::move(device_cols)),
              weights_(std::move(device_weights)),
              vertex_count_(vertex_count),
              edge_count_(edge_count) {}

    /// Pointer to device-resident row offsets (vertex_count + 1 entries).
    const std::int64_t* get_rows() const {
        return rows_.get_data();
    }

    /// Pointer to device-resident column indices (2 * edge_count entries).
    const IndexType* get_cols() const {
        return cols_.get_data();
    }

    /// Pointer to device-resident edge weights (2 * edge_count entries),
    /// or nullptr if the graph is unweighted.
    const WeightType* get_weights() const {
        return weights_.get_count() > 0 ? weights_.get_data() : nullptr;
    }

    std::int64_t get_vertex_count() const {
        return vertex_count_;
    }

    std::int64_t get_edge_count() const {
        return edge_count_;
    }

    /// Returns the underlying device array for row offsets.
    const dal::array<std::int64_t>& get_rows_array() const {
        return rows_;
    }

    /// Returns the underlying device array for column indices.
    const dal::array<IndexType>& get_cols_array() const {
        return cols_;
    }

    /// Returns the underlying device array for edge weights.
    const dal::array<WeightType>& get_weights_array() const {
        return weights_;
    }

private:
    dal::array<std::int64_t> rows_;
    dal::array<IndexType> cols_;
    dal::array<WeightType> weights_;
    std::int64_t vertex_count_ = 0;
    std::int64_t edge_count_ = 0;
};

/// Specialization for unweighted graphs (WeightType = void).
/// This preserves full backward compatibility with existing code.
template <typename IndexType>
class device_csr_topology<IndexType, void> {
public:
    using index_type = IndexType;
    using weight_type = void;

    device_csr_topology() = default;

    device_csr_topology(dal::array<std::int64_t> device_rows,
                        dal::array<IndexType> device_cols,
                        std::int64_t vertex_count,
                        std::int64_t edge_count)
            : rows_(std::move(device_rows)),
              cols_(std::move(device_cols)),
              vertex_count_(vertex_count),
              edge_count_(edge_count) {}

    /// Pointer to device-resident row offsets (vertex_count + 1 entries).
    const std::int64_t* get_rows() const {
        return rows_.get_data();
    }

    /// Pointer to device-resident column indices (2 * edge_count entries).
    const IndexType* get_cols() const {
        return cols_.get_data();
    }

    std::int64_t get_vertex_count() const {
        return vertex_count_;
    }

    std::int64_t get_edge_count() const {
        return edge_count_;
    }

    /// Returns the underlying device array for row offsets.
    const dal::array<std::int64_t>& get_rows_array() const {
        return rows_;
    }

    /// Returns the underlying device array for column indices.
    const dal::array<IndexType>& get_cols_array() const {
        return cols_;
    }

private:
    dal::array<std::int64_t> rows_;
    dal::array<IndexType> cols_;
    std::int64_t vertex_count_ = 0;
    std::int64_t edge_count_ = 0;
};

/// Transfers a host topology to device memory.
/// Row offsets are kept as int64, column indices as IndexType.
///
/// @tparam IndexType  The index type for column indices (default int32)
/// @param queue       SYCL queue identifying the target device
/// @param host_topo   Host-resident CSR topology
/// @return A device_csr_topology with arrays in device USM memory
template <typename IndexType = std::int32_t>
device_csr_topology<IndexType> topology_to_device(sycl::queue& queue,
                                                  const topology<IndexType>& host_topo) {
    const auto vertex_count = host_topo.get_vertex_count();
    const auto edge_count = host_topo.get_edge_count();

    if (vertex_count == 0) {
        return device_csr_topology<IndexType>();
    }

    const std::int64_t rows_count = vertex_count + 1;
    const std::int64_t cols_count = host_topo._cols.get_count();

    // Transfer column indices (IndexType -> IndexType) to device
    dal::array<IndexType> device_cols;
    if (cols_count > 0) {
        device_cols =
            dal::array<IndexType>::empty(queue, cols_count, sycl::usm::alloc::device);
        queue.memcpy(device_cols.get_mutable_data(),
                     host_topo._cols.get_data(),
                     cols_count * sizeof(IndexType))
            .wait_and_throw();
    }

    // Transfer row offsets (int64 -> int64) to device
    auto device_rows =
        dal::array<std::int64_t>::empty(queue, rows_count, sycl::usm::alloc::device);
    queue.memcpy(device_rows.get_mutable_data(),
                 host_topo._rows.get_data(),
                 rows_count * sizeof(std::int64_t))
        .wait_and_throw();

    return device_csr_topology<IndexType>(std::move(device_rows),
                                          std::move(device_cols),
                                          vertex_count,
                                          edge_count);
}

/// Transfers a device_csr_topology back to a host topology.
///
/// @tparam IndexType  The index type for column indices
/// @param device_topo Device-resident CSR topology
/// @return A host topology with arrays in host memory
template <typename IndexType = std::int32_t>
topology<IndexType> topology_to_host(const device_csr_topology<IndexType>& device_topo) {
    const auto vertex_count = device_topo.get_vertex_count();
    const auto edge_count = device_topo.get_edge_count();

    if (vertex_count == 0) {
        return topology<IndexType>();
    }

    const std::int64_t rows_count = vertex_count + 1;
    const std::int64_t cols_count = device_topo.get_cols_array().get_count();

    auto host_cols = dal::array<IndexType>::empty(cols_count);
    auto host_rows = dal::array<std::int64_t>::empty(rows_count);

    // Retrieve the queue from the device arrays
    auto q = device_topo.get_cols_array().get_queue().value();
    q.memcpy(host_cols.get_mutable_data(),
             device_topo.get_cols(),
             cols_count * sizeof(IndexType))
        .wait_and_throw();
    q.memcpy(host_rows.get_mutable_data(),
             device_topo.get_rows(),
             rows_count * sizeof(std::int64_t))
        .wait_and_throw();

    // Build degrees from row offsets
    auto host_degrees = dal::array<IndexType>::empty(vertex_count);
    auto* deg = host_degrees.get_mutable_data();
    const auto* rows_data = host_rows.get_data();
    for (std::int64_t i = 0; i < vertex_count; ++i) {
        deg[i] = static_cast<IndexType>(rows_data[i + 1] - rows_data[i]);
    }

    topology<IndexType> result;
    result._vertex_count = vertex_count;
    result._edge_count = edge_count;
    result._rows = std::move(host_rows);
    result._cols = std::move(host_cols);
    result._degrees = std::move(host_degrees);
    result._rows_ptr = result._rows.get_data();
    result._cols_ptr = result._cols.get_data();
    result._degrees_ptr = result._degrees.get_data();
    return result;
}

} // namespace oneapi::dal::preview::detail

#endif // ONEDAL_DATA_PARALLEL
