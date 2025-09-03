#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/frontier/graph.hpp"
#include "oneapi/dal/backend/primitives/frontier/frontier.hpp"

namespace oneapi::dal::backend::primitives {

/// ContextState holds the state of the current processing group in the kernel
/// It includes the group offset, coarsening factor, offsets size, and the SYCL
/// nd_item for the current work item.
struct ContextState {
    size_t group_offset;
    const uint16_t coarsening_factor;
    const uint32_t offsets_size;
    const sycl::nd_item<1> item;
};

/// Context is a template class that provides an interface for the kernel to
/// interact with the frontier data structure. It holds the limit of the graph,
/// the input frontier, and the output frontier. It provides methods to initialize
/// the context state, check if there are more elements to process, get the
/// assigned element, check if a vertex is in the frontier, and insert a vertex
/// into the output frontier.
template <typename InFrontierDevT, typename OutFrontierDevT>
struct Context {
    size_t limit;
    InFrontierDevT in_dev_frontier;
    OutFrontierDevT out_dev_frontier;

    Context(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
            : limit(limit),
              in_dev_frontier(in_dev_frontier),
              out_dev_frontier(out_dev_frontier) {}

    inline ContextState init(sycl::nd_item<1>& item) const {
        return {
            item.get_group_linear_id(),
            static_cast<uint16_t>(item.get_local_range(0) / in_dev_frontier.get_element_bitsize()),
            in_dev_frontier.get_offsets_size()[0],
            item,
        };
    }

    /// This method checks if there are more elements to process in the current group.
    inline bool need_to_process(ContextState& state) const {
        return (state.group_offset * state.coarsening_factor < state.offsets_size);
    }

    /// This method completes the current iteration by updating the group offset.
    inline void complete_iteration(ContextState& state) const {
        state.group_offset += state.item.get_group_range(0);
    }

    /// This method retrieves the assigned element for the current work item.
    inline size_t get_assigned_element(const ContextState& state) const {
        const uint16_t element_bitsize = in_dev_frontier.get_element_bitsize();
        const uint32_t acutal_id_offset = (state.group_offset * state.coarsening_factor) +
                                          (state.item.get_local_linear_id() / element_bitsize);
        const uint32_t* bitmap_offsets = in_dev_frontier.get_offsets();
        const auto assigned_vertex = (bitmap_offsets[acutal_id_offset] * element_bitsize) +
                                     (state.item.get_local_linear_id() % element_bitsize);
        return assigned_vertex;
    }

    /// This method checks if a vertex is in the input frontier.
    inline bool check(const ContextState& state, size_t vertex) const {
        return vertex < limit && in_dev_frontier.check(vertex);
    }

    /// This method inserts a vertex into the output frontier.
    inline void insert(const ContextState& state, size_t vertex) const {
        out_dev_frontier.insert(vertex);
    }
};

/// BitmapKernel is a template struct that defines the kernel to be executed on the device.
/// It processes the input frontier and advances it based on the provided functor.
template <typename T, typename ContextT, typename GraphDevT, typename LambdaT>
struct BitmapKernel {
    void operator()(sycl::nd_item<1> item) const {
        const size_t lid = item.get_local_linear_id();
        const auto wgroup = item.get_group();
        const size_t wgroup_size = wgroup.get_local_range(0);
        const auto sgroup = item.get_sub_group();
        const auto sgroup_id = sgroup.get_group_id();
        const size_t sgroup_size = sgroup.get_local_range()[0];
        const size_t llid = sgroup.get_local_linear_id();

        auto state = context.init(item);

        while (context.need_to_process(state)) {
            const auto assigned_vertex = context.get_assigned_element(state);

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

            // 1. distribute the workload based on the number of edges to the workgroup, subgroup or workitem
            const uint32_t offset = sgroup_id * sgroup_size;
            if (context.check(state, assigned_vertex)) {
                uint32_t n_edges = graph_dev.get_degree(assigned_vertex);
                if (n_edges >=
                    wgroup_size *
                        wgroup_size) { // if the number of edges is large enough, we can assign the vertex to the workgroup
                    uint32_t loc = wg_tail.fetch_add(static_cast<uint32_t>(1));
                    n_edges_wg[loc] = n_edges;
                    workgroup_reduce[loc] = assigned_vertex;
                    workgroup_ids[loc] = lid;
                }
                else if (n_edges >= sgroup_size) { // assign to the subgroup
                    uint32_t loc = sg_tail.fetch_add(static_cast<uint32_t>(1));
                    n_edges_sg[offset + loc] = n_edges;
                    subgroup_reduce[offset + loc] = assigned_vertex;
                    subgroup_ids[offset + loc] = lid;
                }
                visited[lid] = false;
            }
            else {
                visited[lid] = true;
            }

            sycl::group_barrier(wgroup);

            // 2. process the the edge of one vertex at a time using the workgroup
            for (size_t i = 0; i < wg_tail.load(); i++) {
                auto vertex = workgroup_reduce[i];
                size_t n_edges = n_edges_wg[i];
                auto start = graph_dev.begin(vertex);

                for (auto j = lid; j < n_edges; j += wgroup_size) {
                    auto n = start + j;
                    auto edge = n.get_index();
                    auto weight = graph_dev.get_weight(edge);
                    auto neighbor = *n;
                    if (functor(vertex, neighbor, edge, weight)) {
                        context.insert(state, neighbor);
                    }
                }

                if (wgroup.leader()) {
                    visited[workgroup_ids[i]] = true;
                }
            }

            sycl::group_barrier(sgroup);

            // 3. process the edges of one vertex at a time using the subgroup
            for (
                size_t i = 0; i < subgroup_reduce_tail[sgroup_id];
                i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
                size_t vertex_id = offset + i;
                auto vertex = subgroup_reduce[vertex_id];
                size_t n_edges = n_edges_sg[vertex_id];

                auto start = graph_dev.begin(vertex);

                for (auto j = llid; j < n_edges; j += sgroup_size) {
                    auto n = start + j;
                    auto edge = n.get_index();
                    auto weight = graph_dev.get_weight(edge);
                    auto neighbor = *n;
                    if (functor(vertex, neighbor, edge, weight)) {
                        context.insert(state, neighbor);
                    }
                }

                if (sgroup.leader()) {
                    visited[subgroup_ids[vertex_id]] = true;
                }
            }
            sycl::group_barrier(sgroup);

            // 4. process the edges of the vertices that were not assigned to the workgroup or subgroup
            if (!visited[lid]) {
                auto vertex = assigned_vertex;
                auto start = graph_dev.begin(vertex);
                auto end = graph_dev.end(vertex);

                for (auto n = start; n != end; ++n) {
                    auto edge = n.get_index();
                    auto weight = graph_dev.get_weight(edge);
                    auto neighbor = *n;
                    if (functor(vertex, neighbor, edge, weight)) {
                        context.insert(state, neighbor);
                    }
                }
            }
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

/// advance is a function that launches the BitmapKernel on the device to advance the frontier.
/// It takes the graph, input frontier, output frontier, and the functor.
/// The expected_size parameter is used to specify the expected size of the input frontier, in order
/// to optimize the kernel launch parameters. If expected_size is zero, a default size is used.
/// It returns a sycl::event that can be used to synchronize the execution.
template <typename FrontierSizeT, typename GraphT, typename LambdaT>
sycl::event advance(const GraphT& graph,
                    frontier<FrontierSizeT>& in,
                    frontier<FrontierSizeT>& out,
                    LambdaT&& functor) {
    using element_t = FrontierSizeT;

    sycl::queue& q = graph.get_queue();

    size_t num_nodes = graph.get_vertex_count();

    auto in_dev_frontier = in.get_device_view();
    auto out_dev_frontier = out.get_device_view();
    auto graph_dev = graph.get_device_view();

    /// The coarsening factor represents the number of integers a single work-group should handle.
    size_t coarsening_factor = 1; // it should be Compute Unit Size / num subgroups;

    size_t element_bitsize = in_dev_frontier.get_element_bitsize(); // in bits
    sycl::range<1> local_range = { element_bitsize * coarsening_factor };
    size_t global_size;

    sycl::event to_wait =
        in.compute_active_frontier(); // kernel launch to compute active bitmap regions
    auto offset_size = in.get_offsets_size().at_device(q, 0, { to_wait });
    global_size = offset_size * element_bitsize;

    sycl::range<1> global_range{ (global_size % local_range[0] == 0)
                                     ? global_size
                                     : global_size +
                                           (local_range[0] - global_size % local_range[0]) };

    Context<decltype(in_dev_frontier), decltype(out_dev_frontier)> context{ num_nodes,
                                                                            in_dev_frontier,
                                                                            out_dev_frontier };
    using bitmap_kernel_t =
        BitmapKernel<element_t, decltype(context), decltype(graph_dev), LambdaT>;

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
