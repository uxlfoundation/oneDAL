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

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/shortest_paths/backend/gpu/shortest_paths.hpp"
#include "oneapi/dal/algo/shortest_paths/backend/gpu/traverse_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/backend/primitives/frontier.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {

namespace fp = oneapi::dal::preview::backend::primitives;

namespace detail_gpu {

/// Device-view for weighted CSR with int64 row offsets + int32 column indices.
/// Compatible with the frontier advance kernel API (provides get_degree,
/// get_weight, begin, end).
template <typename Index, typename OffsetT = std::int64_t>
class weighted_csr_graph_view {
public:
    struct neighbor_iterator_t {
        neighbor_iterator_t(const Index* start_ptr, Index* ptr)
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
        SYCL_EXTERNAL Index operator*() const {
            return *_ptr;
        }
        SYCL_EXTERNAL std::uint32_t get_index() const {
            return static_cast<std::uint32_t>(_ptr - _start_ptr);
        }

        Index* _ptr;
        const Index* _start_ptr;
    };

    weighted_csr_graph_view(std::uint64_t num_nodes,
                            const OffsetT* row_ptr,
                            Index* col_indices,
                            const void* weights_ptr)
            : _num_nodes(num_nodes),
              _row_ptr(row_ptr),
              _col_indices(col_indices),
              _weights_ptr(weights_ptr) {}

    SYCL_EXTERNAL inline std::uint32_t get_degree(const std::uint32_t vertex) const {
        return static_cast<std::uint32_t>(_row_ptr[vertex + 1] - _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline std::uint32_t get_weight(const std::uint32_t) const {
        // Weight is accessed directly in the functor via the weights pointer;
        // this method satisfies the advance API interface requirement.
        return 0;
    }

    SYCL_EXTERNAL inline neighbor_iterator_t begin(std::uint32_t vertex) const {
        return neighbor_iterator_t(_col_indices,
                                   _col_indices + _row_ptr[vertex]);
    }

    SYCL_EXTERNAL inline neighbor_iterator_t end(std::uint32_t vertex) const {
        return neighbor_iterator_t(_col_indices,
                                   _col_indices + _row_ptr[vertex + 1]);
    }

private:
    std::uint64_t _num_nodes;
    const OffsetT* _row_ptr;
    Index* _col_indices;
    const void* _weights_ptr;
};

/// Device wrapper for weighted CSR graph compatible with frontier advance.
template <typename Index, typename Weight, typename OffsetT = std::int64_t>
class weighted_csr_graph_device_wrapper {
public:
    weighted_csr_graph_device_wrapper(sycl::queue& queue,
                                      std::uint64_t num_nodes,
                                      const OffsetT* row_ptr,
                                      Index* col_indices,
                                      const Weight* weights)
            : _queue(queue),
              _num_nodes(num_nodes),
              _row_ptr(row_ptr),
              _col_indices(col_indices),
              _weights(weights) {}

    weighted_csr_graph_view<Index, OffsetT> get_device_view() const {
        return weighted_csr_graph_view<Index, OffsetT>(
            _num_nodes,
            _row_ptr,
            const_cast<Index*>(_col_indices),
            _weights);
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
    const OffsetT* _row_ptr;
    Index* _col_indices;
    const Weight* _weights;
};

/// Computes single-source shortest paths on the GPU using a frontier-driven
/// label-correcting approach with the oneDAL frontier advance primitive.
template <typename Index, typename Weight>
void compute_sssp_gpu(sycl::queue& queue,
                      const std::int64_t* rows,
                      const Index* cols,
                      const Weight* weights,
                      std::int64_t vertex_count,
                      std::int64_t edge_count,
                      std::int64_t source,
                      bool compute_predecessors,
                      Weight* distances,
                      std::int32_t* predecessors) {
    constexpr Weight inf = std::numeric_limits<Weight>::max();

    // Initialize distances to infinity, source to 0
    queue
        .submit([&](sycl::handler& cgh) {
            Weight* d_dist = distances;
            const std::int64_t vc = vertex_count;
            const std::int64_t src = source;
            cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
                d_dist[idx[0]] = (static_cast<std::int64_t>(idx[0]) == src)
                                     ? Weight{ 0 }
                                     : inf;
            });
        })
        .wait_and_throw();

    // Initialize predecessors to -1 if requested
    if (compute_predecessors && predecessors != nullptr) {
        queue.memset(predecessors, 0xFF, vertex_count * sizeof(std::int32_t)).wait_and_throw();
    }

    // Fast path: if no edges, only source has distance 0
    if (edge_count == 0) {
        return;
    }

    // Create device graph wrapper for frontier advance
    weighted_csr_graph_device_wrapper<Index, Weight, std::int64_t> graph(
        queue,
        static_cast<std::uint64_t>(vertex_count),
        rows,
        const_cast<Index*>(cols),
        weights);

    // Create two-layer bitmap frontiers for double-buffering
    fp::frontier<std::uint32_t> in_frontier(queue, vertex_count, sycl::usm::alloc::device);
    fp::frontier<std::uint32_t> out_frontier(queue, vertex_count, sycl::usm::alloc::device);

    // Initialize: only source vertex is active
    {
        auto fv = in_frontier.get_device_view();
        const std::uint32_t src32 = static_cast<std::uint32_t>(source);
        queue
            .submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                    fv.insert(src32);
                });
            })
            .wait_and_throw();
    }

    // Label-correcting SSSP with workload-balanced frontier advance
    Weight* d_dist = distances;
    std::int32_t* d_pred = predecessors;
    const bool track_pred = compute_predecessors && (predecessors != nullptr);
    const Weight* d_weights = weights;

    while (!in_frontier.empty()) {
        auto e = fp::advance(
            graph,
            in_frontier,
            out_frontier,
            [=](auto src_v, auto dst, auto edge, auto) -> bool {
                const Weight dist_src = d_dist[src_v];

                if (dist_src == inf || d_weights[edge] >= inf - dist_src) {
                    return false;
                }

                const Weight new_dist = dist_src + d_weights[edge];
                const Weight old_dist =
                    sycl::atomic_ref<Weight,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(d_dist[dst])
                        .fetch_min(new_dist);

                if (new_dist < old_dist) {
                    if (track_pred) {
                        d_pred[dst] = static_cast<std::int32_t>(src_v);
                    }
                    return true;
                }
                return false;
            });
        e.wait_and_throw();

        fp::swap_frontiers(in_frontier, out_frontier);
        out_frontier.clear();
    }
}

} // namespace detail_gpu

template <typename Float, typename Task, typename Index, typename Weight>
traverse_result<Task> run_shortest_paths_gpu(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const csr_topology_gpu_view<Index, Weight>& topology) {
    auto& queue = ctx.get_queue();

    const auto vertex_count = topology.vertex_count;
    const auto edge_count = topology.edge_count;
    const auto source = desc.get_source();

    if (vertex_count == 0) {
        return traverse_result<Task>();
    }

    const bool need_predecessors =
        static_cast<bool>(desc.get_optional_results() & optional_results::predecessors);
    const bool need_distances =
        static_cast<bool>(desc.get_optional_results() & optional_results::distances);

    // Allocate output arrays in shared memory
    auto* distances = sycl::malloc_shared<Weight>(vertex_count, queue);
    std::int32_t* predecessors = nullptr;
    if (need_predecessors) {
        predecessors = sycl::malloc_shared<std::int32_t>(vertex_count, queue);
    }

    detail_gpu::compute_sssp_gpu(queue,
                                 topology.rows,
                                 topology.cols,
                                 topology.weights,
                                 vertex_count,
                                 edge_count,
                                 source,
                                 need_predecessors,
                                 distances,
                                 predecessors);

    traverse_result<Task> result;

    if (need_distances) {
        auto dist_arr =
            array<Weight>(queue, distances, vertex_count, [queue](Weight* ptr) mutable {
                sycl::free(ptr, queue);
            });
        result.set_distances(
            dal::detail::homogen_table_builder{}
                .reset(dist_arr, vertex_count, 1)
                .build());
    }
    else {
        sycl::free(distances, queue);
    }

    if (need_predecessors && predecessors != nullptr) {
        auto pred_arr = array<std::int32_t>(
            queue,
            predecessors,
            vertex_count,
            [queue](std::int32_t* ptr) mutable { sycl::free(ptr, queue); });
        result.set_predecessors(
            dal::detail::homogen_table_builder{}
                .reset(pred_arr, vertex_count, 1)
                .build());
    }

    return result;
}

template <typename Float, typename Task, typename Topology, typename EdgeValue>
traverse_result<Task> traverse_kernel_gpu<Float, Task, Topology, EdgeValue>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const Topology& t,
    const EdgeValue* edge_values) const {
    using index_type = typename Topology::vertex_type;
    auto& queue = ctx.get_queue();

    const auto vertex_count = t.get_vertex_count();
    const auto edge_count = t.get_edge_count();

    if (vertex_count == 0) {
        return traverse_result<Task>();
    }

    // Check if topology data is already on device
    const auto* rows_ptr = t._rows.get_data();
    const auto ptr_type = sycl::get_pointer_type(rows_ptr, queue.get_context());

    const std::int64_t* device_rows = nullptr;
    const index_type* device_cols = nullptr;
    dal::preview::detail::device_csr_topology<index_type> device_topo_holder;

    if (ptr_type == sycl::usm::alloc::device || ptr_type == sycl::usm::alloc::shared) {
        device_rows = t._rows.get_data();
        device_cols = t._cols.get_data();
    }
    else {
        device_topo_holder =
            dal::preview::detail::topology_to_device<index_type>(queue, t);
        device_rows = device_topo_holder.get_rows();
        device_cols = device_topo_holder.get_cols();
    }

    // Transfer edge weights to device
    const std::int64_t weights_count = t._cols.get_count();
    dal::array<EdgeValue> device_weights;
    EdgeValue* device_weights_ptr = nullptr;

    if (weights_count > 0 && edge_values != nullptr) {
        const auto weights_ptr_type =
            sycl::get_pointer_type(edge_values, queue.get_context());
        if (weights_ptr_type == sycl::usm::alloc::device ||
            weights_ptr_type == sycl::usm::alloc::shared) {
            device_weights_ptr = const_cast<EdgeValue*>(edge_values);
        }
        else {
            auto* buf = sycl::malloc_shared<EdgeValue>(weights_count, queue);
            queue.memcpy(buf, edge_values, weights_count * sizeof(EdgeValue))
                .wait_and_throw();
            device_weights = dal::array<EdgeValue>(
                queue, buf, weights_count, [queue](EdgeValue* p) mutable {
                    sycl::free(p, queue);
                });
            device_weights_ptr = const_cast<EdgeValue*>(device_weights.get_data());
        }
    }

    csr_topology_gpu_view<index_type, EdgeValue> gpu_view;
    gpu_view.rows = device_rows;
    gpu_view.cols = device_cols;
    gpu_view.weights = device_weights_ptr;
    gpu_view.vertex_count = vertex_count;
    gpu_view.edge_count = edge_count;

    return run_shortest_paths_gpu<Float, Task>(ctx, desc, gpu_view);
}

// Explicit instantiations
template struct traverse_kernel_gpu<float,
                                    task::one_to_all,
                                    dal::preview::detail::topology<std::int32_t>,
                                    std::int32_t>;

template struct traverse_kernel_gpu<float,
                                    task::one_to_all,
                                    dal::preview::detail::topology<std::int32_t>,
                                    double>;

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, std::int32_t, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t, std::int32_t>&);

template traverse_result<task::one_to_all>
run_shortest_paths_gpu<float, task::one_to_all, std::int32_t, double>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::one_to_all>&,
    const csr_topology_gpu_view<std::int32_t, double>&);

} // namespace oneapi::dal::preview::shortest_paths::backend
