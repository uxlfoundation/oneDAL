/*******************************************************************************
* Copyright 2025 University of Salerno
* testtesttest
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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/frontier/graph.hpp"
#include "oneapi/dal/backend/primitives/frontier/frontier.hpp"

namespace oneapi::dal::backend::primitives {

/// frontier_context_state holds the state of the current processing group in the kernel
/// It includes the group offset, coarsening factor, offsets size, and the SYCL
/// nd_item for the current work item.
struct frontier_context_state {
    std::uint64_t group_offset; // Offset of the current group being processed
    const uint16_t coarsening_factor; // Number of integers a single work-group should handle
    const uint32_t offsets_size; // Size of the offsets buffer array
    const sycl::nd_item<1> item;
};

/// frontier_context is a template class that provides an interface for the kernel to
/// interact with the frontier data structure. It holds the limit of the graph,
/// the input frontier, and the output frontier. It provides methods to initialize
/// the context state, check if there are more elements to process, get the
/// assigned element, check if a vertex is in the frontier, and insert a vertex
/// into the output frontier.
/// \tparam InFrontierDevT input frontier device view type. Inferred from the input frontier.
/// \tparam OutFrontierDevT output frontier device view type. Inferred from the output frontier.
template <typename InFrontierDevT, typename OutFrontierDevT>
class frontier_context {
public:
    frontier_context(std::uint64_t limit,
                     InFrontierDevT in_dev_frontier,
                     OutFrontierDevT out_dev_frontier)
            : limit(limit),
              in_dev_frontier(in_dev_frontier),
              out_dev_frontier(out_dev_frontier) {}

    inline frontier_context_state init(sycl::nd_item<1>& item) const {
        return {
            item.get_group_linear_id(),
            static_cast<uint16_t>(item.get_local_range(0) / in_dev_frontier.get_element_bitsize()),
            in_dev_frontier.get_offsets_size()[0],
            item,
        };
    }

    /// Checks if there are more elements to process in the current group.
    inline bool need_to_process(frontier_context_state& state) const {
        return (state.group_offset * state.coarsening_factor < state.offsets_size);
    }

    /// Completes the current iteration by updating the group offset.
    inline void complete_iteration(frontier_context_state& state) const {
        state.group_offset += state.item.get_group_range(0);
    }

    /// Retrieves the assigned element for the current work item.
    inline std::uint64_t get_assigned_element(const frontier_context_state& state) const {
        const uint16_t element_bitsize = in_dev_frontier.get_element_bitsize();
        const uint32_t actual_id_offset = (state.group_offset * state.coarsening_factor) +
                                          (state.item.get_local_linear_id() / element_bitsize);
        ONEDAL_ASSERT(actual_id_offset < in_dev_frontier.get_offsets_size()[0]);
        const uint32_t* bitmap_offsets = in_dev_frontier.get_offsets();
        const auto assigned_vertex = (bitmap_offsets[actual_id_offset] * element_bitsize) +
                                     (state.item.get_local_linear_id() % element_bitsize);
        return assigned_vertex;
    }

    /// Checks if a vertex is in the input frontier.
    inline bool check(std::uint64_t vertex) const {
        return vertex < limit && in_dev_frontier.check(vertex);
    }

    /// Inserts a vertex into the output frontier.
    inline void insert(std::uint64_t vertex) const {
        out_dev_frontier.insert(vertex);
    }

    std::uint64_t limit;
    InFrontierDevT in_dev_frontier;
    OutFrontierDevT out_dev_frontier;
};

/// bitmap_kernel is a template struct that defines the kernel to be executed on the device.
/// It processes the input frontier and advances it based on the provided functor.
/// \tparam T the type of the elements in the frontier (e.g., std::uint32_t).
/// \tparam ContextT the type of the frontier context. Inferred from the frontier_context.
/// \tparam GraphDevT the type of the graph device view. Inferred from the graph.
/// \tparam LambdaT the type of the functor to be applied to each edge.
template <typename T, typename ContextT, typename GraphDevT, typename LambdaT>
struct bitmap_kernel {
    /// Distributes the workload among the workgroup, subgroup, and individual work items
    template <typename VertexT>
    inline void distribute_workload(frontier_context_state& state, const VertexT& vertex) const {
        const std::uint64_t lid = state.item.get_local_linear_id();
        const auto wgroup = state.item.get_group();
        const std::uint64_t wgroup_size = wgroup.get_local_range(0);
        const auto sgroup = state.item.get_sub_group();
        const auto sgroup_id = sgroup.get_group_id();
        const std::uint64_t sgroup_size = sgroup.get_local_range()[0];

        if (sgroup.leader()) {
            subgroup_reduce_tail[sgroup_id] = 0;
        }
        if (wgroup.leader()) {
            workgroup_reduce_tail[0] = 0;
        }

        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group>
            sg_tail{ subgroup_reduce_tail[sgroup_id] };
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>
            wg_tail{ workgroup_reduce_tail[0] };
        const uint32_t offset = sgroup_id * sgroup_size;

        if (context.check(vertex)) {
            uint32_t n_edges = graph_dev.get_degree(vertex);
            // if the number of edges is large enough, we can assign the vertex to the workgroup
            if (n_edges >= wgroup_size * wgroup_size) {
                uint32_t loc = wg_tail.fetch_add(static_cast<uint32_t>(1));
                n_edges_wg[loc] = n_edges;
                workgroup_reduce[loc] = vertex;
                workgroup_ids[loc] = lid;
            }
            else if (n_edges >= sgroup_size) { // assign to the subgroup
                uint32_t loc = sg_tail.fetch_add(static_cast<uint32_t>(1));
                n_edges_sg[offset + loc] = n_edges;
                subgroup_reduce[offset + loc] = vertex;
                subgroup_ids[offset + loc] = lid;
            }
            visited[lid] = false;
        }
        else {
            visited[lid] = true;
        }
    }

    /// Processes the vertices assigned to the workgroup.
    inline void process_workgroup_reduction(frontier_context_state& state) const {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>
            wg_tail{ workgroup_reduce_tail[0] };

        const auto wgroup = state.item.get_group();
        const std::uint64_t lid = state.item.get_local_linear_id();
        const std::uint64_t wgroup_size = wgroup.get_local_range(0);

        for (std::uint64_t i = 0; i < wg_tail.load(); i++) {
            auto vertex = workgroup_reduce[i];
            std::uint64_t n_edges = n_edges_wg[i];
            auto start = graph_dev.begin(vertex);

            for (auto j = lid; j < n_edges; j += wgroup_size) {
                auto n = start + j;
                auto edge = n.get_index();
                auto weight = graph_dev.get_weight(edge);
                auto neighbor = *n;
                if (functor(vertex, neighbor, edge, weight)) {
                    context.insert(neighbor);
                }
            }

            if (wgroup.leader()) {
                visited[workgroup_ids[i]] = true;
            }
        }
    }

    /// Processes the vertices assigned to the subgroup.
    inline void process_subgroup_reduction(frontier_context_state& state) const {
        const auto sgroup = state.item.get_sub_group();
        const auto sgroup_id = sgroup.get_group_id();
        const std::uint64_t sgroup_size = sgroup.get_local_range()[0];
        const std::uint64_t llid = sgroup.get_local_linear_id();
        const uint32_t offset = sgroup_id * sgroup_size;

        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group>
            sg_tail{ subgroup_reduce_tail[sgroup_id] };

        for (std::uint64_t i = 0; i < subgroup_reduce_tail[sgroup_id]; i++) {
            // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
            std::uint64_t vertex_id = offset + i;
            auto vertex = subgroup_reduce[vertex_id];
            std::uint64_t n_edges = n_edges_sg[vertex_id];

            auto start = graph_dev.begin(vertex);

            for (auto j = llid; j < n_edges; j += sgroup_size) {
                auto n = start + j;
                auto edge = n.get_index();
                auto weight = graph_dev.get_weight(edge);
                auto neighbor = *n;
                if (functor(vertex, neighbor, edge, weight)) {
                    context.insert(neighbor);
                }
            }

            if (sgroup.leader()) {
                visited[subgroup_ids[vertex_id]] = true;
            }
        }
    }

    /// Processes the vertices that were not assigned to the workgroup or subgroup.
    /// \tparam VertexT the type of the vertex. Inferred from the graph.
    template <typename VertexT>
    inline void process_workitem_reduction(frontier_context_state& state,
                                           const VertexT& vertex) const {
        const std::uint64_t lid = state.item.get_local_linear_id();

        if (!visited[lid]) {
            auto start = graph_dev.begin(vertex);
            auto end = graph_dev.end(vertex);

            for (auto n = start; n != end; ++n) {
                auto edge = n.get_index();
                auto weight = graph_dev.get_weight(edge);
                auto neighbor = *n;
                if (functor(vertex, neighbor, edge, weight)) {
                    context.insert(neighbor);
                }
            }
        }
    }

    /// The main operator that executes the kernel logic.
    void operator()(sycl::nd_item<1> item) const {
        auto state = context.init(item);
        const auto wgroup = item.get_group();
        const auto sgroup = item.get_sub_group();

        while (context.need_to_process(state)) {
            const auto assigned_vertex = context.get_assigned_element(state);

            // 1. distribute the workload based on the number of edges to the workgroup, subgroup or workitem
            distribute_workload(state, assigned_vertex);
            sycl::group_barrier(wgroup);

            // 2. process the the edge of one vertex at a time using the workgroup
            process_workgroup_reduction(state);
            sycl::group_barrier(sgroup);

            // 3. process the edges of one vertex at a time using the subgroup
            process_subgroup_reduction(state);
            sycl::group_barrier(sgroup);

            // 4. process the edges of the vertices that were not assigned to the workgroup or subgroup
            process_workitem_reduction(state, assigned_vertex);

            context.complete_iteration(state); // complete the current iteration
        }
    }

    const ContextT context;
    const GraphDevT graph_dev;
    const sycl::local_accessor<uint32_t, 1> n_edges_wg;
    const sycl::local_accessor<uint32_t, 1> n_edges_sg;
    const sycl::local_accessor<bool, 1> visited;
    const sycl::local_accessor<T, 1> subgroup_reduce;
    const sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail;
    const sycl::local_accessor<uint32_t, 1> subgroup_ids;
    const sycl::local_accessor<T, 1> workgroup_reduce;
    const sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail;
    const sycl::local_accessor<uint32_t, 1> workgroup_ids;
    const LambdaT functor;
};

/// Advance is a function that launches the bitmap_kernel on the device to advance the frontier.
/// It takes the graph, input frontier, output frontier, and the functor.
/// It returns a sycl::event that can be used to synchronize the execution.
/// \tparam FrontierSizeT the type of the elements in the frontier (e.g., std::uint32_t).
/// \tparam GraphT the type of the graph. Inferred from the csr_graph.
/// \tparam LambdaT the type of the functor to be applied to each edge.
template <typename FrontierSizeT, typename GraphT, typename LambdaT>
sycl::event advance(const GraphT& graph,
                    frontier<FrontierSizeT>& in,
                    frontier<FrontierSizeT>& out,
                    LambdaT&& functor) {
    using element_t = FrontierSizeT;

    sycl::queue& q = graph.get_queue();

    std::uint64_t num_nodes = graph.get_vertex_count();

    auto in_dev_frontier = in.get_device_view();
    auto out_dev_frontier = out.get_device_view();
    auto graph_dev = graph.get_device_view();

    /// The coarsening factor represents the number of integers a single work-group should handle.
    std::uint64_t coarsening_factor = 1; // it should be Compute Unit Size / num subgroups;

    std::uint64_t element_bitsize = in_dev_frontier.get_element_bitsize(); // in bits
    sycl::range<1> local_range = { element_bitsize * coarsening_factor };
    std::uint64_t global_size;

    sycl::event to_wait =
        in.compute_active_frontier(); // kernel launch to compute active bitmap regions
    auto offset_size = in.get_offsets_size().at_device(q, 0, { to_wait });
    global_size = offset_size * element_bitsize;

    sycl::range<1> global_range{ (global_size % local_range[0] == 0)
                                     ? global_size
                                     : global_size +
                                           (local_range[0] - global_size % local_range[0]) };

    frontier_context<decltype(in_dev_frontier), decltype(out_dev_frontier)> context{
        num_nodes,
        in_dev_frontier,
        out_dev_frontier
    };
    using bitmap_kernel_t =
        bitmap_kernel<element_t, decltype(context), decltype(graph_dev), LambdaT>;

    const uint32_t max_num_subgroups = device_max_sg_count(q);

    auto e = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(to_wait);
        sycl::local_accessor<uint32_t, 1> n_edges_wg{
            local_range,
            cgh
        }; // number of edges of vertices to process at work-group granularity
        sycl::local_accessor<uint32_t, 1> n_edges_sg{
            local_range,
            cgh
        }; // number of edges of vertices to process at sub-group granularity
        sycl::local_accessor<bool, 1> visited{
            local_range,
            cgh
        }; // tracks the vertices already visited at work-group and sub-group granularity
        sycl::local_accessor<element_t, 1> subgroup_reduce{
            local_range,
            cgh
        }; // stores the vertices to process at sub-group granularity
        sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{
            max_num_subgroups,
            cgh
        }; // stores the tail of the subgroup reduction
        sycl::local_accessor<uint32_t, 1> subgroup_ids{
            local_range,
            cgh
        }; // stores the thread id that found the vertex to process at sub-group level
        sycl::local_accessor<element_t, 1> workgroup_reduce{
            local_range,
            cgh
        }; // stores the vertices to process at work-group granularity
        sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail{
            1,
            cgh
        }; // stores the tail of the work-group reduction
        sycl::local_accessor<uint32_t, 1> workgroup_ids{
            local_range,
            cgh
        }; // stores the thread id that found the vertex to process at work-group level
        sycl::local_accessor<element_t, 1> individual_reduce{
            local_range,
            cgh
        }; // stores the vertices to process at individual thread granularity
        sycl::local_accessor<uint32_t, 1> individual_reduce_tail{
            1,
            cgh
        }; // stores the tail of the individual thread reduction

        cgh.parallel_for(sycl::nd_range<1>{ global_range, local_range },
                         bitmap_kernel_t{ context,
                                          graph_dev,
                                          n_edges_wg,
                                          n_edges_sg,
                                          visited,
                                          subgroup_reduce,
                                          subgroup_reduce_tail,
                                          subgroup_ids,
                                          workgroup_reduce,
                                          workgroup_reduce_tail,
                                          workgroup_ids,
                                          std::forward<LambdaT>(functor) });
    });
    return { e };
}

} // namespace oneapi::dal::backend::primitives
