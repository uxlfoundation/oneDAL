#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/frontier/graph.hpp"
#include "oneapi/dal/backend/primitives/frontier/frontier_dpc.hpp"


namespace oneapi::dal::backend::primitives {

struct ContextState {
  size_t group_offset;
  const uint16_t coarsening_factor;
  const uint32_t offsets_size;
  const sycl::nd_item<1> item;
};

template<typename InFrontierDevT, typename OutFrontierDevT>
struct Context {
  size_t limit;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;

  Context(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
      : limit(limit), in_dev_frontier(in_dev_frontier), out_dev_frontier(out_dev_frontier) {}

  inline ContextState init(sycl::nd_item<1>& item) const {
    return {
        item.get_group_linear_id(),
        static_cast<uint16_t>(item.get_local_range(0) / in_dev_frontier.get_element_bitsize()),
        in_dev_frontier.get_offsets_size()[0],
        item,
    };
  }

  inline bool need_to_process(ContextState& state) const { return (state.group_offset * state.coarsening_factor < state.offsets_size); }

  inline void complete_iteration(ContextState& state) const { state.group_offset += state.item.get_group_range(0); }

  inline size_t get_assigned_element(const ContextState& state) const {
    const uint16_t bitmap_range = in_dev_frontier.get_element_bitsize();
    const uint32_t acutal_id_offset = (state.group_offset * state.coarsening_factor) + (state.item.get_local_linear_id() / bitmap_range);
    const auto* bitmap_offsets = in_dev_frontier.get_offsets();
    // const size_t acutal_id_offset = (item.get_group_linear_id() * coarsening_factor) + (item.get_local_linear_id() / bitmap_range);
    const auto assigned_vertex = (bitmap_offsets[acutal_id_offset] * bitmap_range) + (state.item.get_local_linear_id() % bitmap_range);
    return assigned_vertex;
  }

  inline bool check(const ContextState& state, size_t vertex) const {
    return vertex < limit && in_dev_frontier.check(vertex);
    
  }

  inline void insert(const ContextState& state, size_t vertex) const {
    out_dev_frontier.insert(vertex);
  }
};

template<typename T,
         typename ContextT,
         typename GraphDevT,
         typename LambdaT>
struct BitmapKernel {
  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    // const size_t gid = item.get_global_linear_id();
    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    // const auto wgroup_id = item.get_group_linear_id();
    const size_t wgroup_size = wgroup.get_local_range(0);
    const auto sgroup = item.get_sub_group();
    const auto sgroup_id = sgroup.get_group_id();
    const size_t sgroup_size = sgroup.get_local_range()[0];
    const size_t llid = sgroup.get_local_linear_id();

    auto state = context.init(item);


    while (context.need_to_process(state)) {
      const auto assigned_vertex = context.get_assigned_element(state);

      // 1. load number of edges in local memory
      if (sgroup.leader()) { subgroup_reduce_tail[sgroup_id] = 0; }
      if (wgroup.leader()) { workgroup_reduce_tail[0] = 0; }

      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{subgroup_reduce_tail[sgroup_id]};
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{workgroup_reduce_tail[0]};

      const uint32_t offset = sgroup_id * sgroup_size;
      if (context.check(state, assigned_vertex)) {
        uint32_t n_edges = graph_dev.get_degree(assigned_vertex); // GRAPH degree of the vertex
        if (n_edges >= wgroup_size * wgroup_size) { // assign to the workgroup
          uint32_t loc = wg_tail.fetch_add(static_cast<uint32_t>(1));
          n_edges_wg[loc] = n_edges;
          workgroup_reduce[loc] = assigned_vertex;
          workgroup_ids[loc] = lid;
        } else if (n_edges >= sgroup_size) { // assign to the subgroup
          uint32_t loc = sg_tail.fetch_add(static_cast<uint32_t>(1));
          n_edges_sg[offset + loc] = n_edges;
          subgroup_reduce[offset + loc] = assigned_vertex;
          subgroup_ids[offset + loc] = lid;
        }
        visited[lid] = false;
      } else {
        visited[lid] = true;
      }

      sycl::group_barrier(wgroup);
      for (size_t i = 0; i < wg_tail.load(); i++) {
        auto vertex = workgroup_reduce[i];
        size_t n_edges = n_edges_wg[i];
        auto start = graph_dev.begin(vertex); // GRAPH edge iterator

        for (auto j = lid; j < n_edges; j += wgroup_size) {
          auto n = start + j;
          auto edge = n.get_index(); // GRAPH edge index
          auto weight = graph_dev.get_weight(edge); // GRAPH edge weight
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) { context.insert(state, neighbor); }
        }

        if (wgroup.leader()) { visited[workgroup_ids[i]] = true; }
      }

      sycl::group_barrier(sgroup);

      for (size_t i = 0; i < subgroup_reduce_tail[sgroup_id]; i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
        size_t vertex_id = offset + i;
        auto vertex = subgroup_reduce[vertex_id];
        size_t n_edges = n_edges_sg[vertex_id];

        auto start = graph_dev.begin(vertex); // GRAPH edge iterator

        for (auto j = llid; j < n_edges; j += sgroup_size) {
          auto n = start + j;
          auto edge = n.get_index(); // GRAPH edge index
          auto weight = graph_dev.get_weight(edge); // GRAPH edge weight
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) { context.insert(state, neighbor); }
        }

        if (sgroup.leader()) { visited[subgroup_ids[vertex_id]] = true; }
      }
      sycl::group_barrier(sgroup);

      if (!visited[lid]) {
        auto vertex = assigned_vertex;
        auto start = graph_dev.begin(vertex); // GRAPH edge iterator
        auto end = graph_dev.end(vertex); // GRAPH edge iterator

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index(); // GRAPH edge index
          auto weight = graph_dev.get_weight(edge); // GRAPH edge weight
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) { context.insert(state, neighbor); }
        }
      }
      context.complete_iteration(state);
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

template<typename FrontierSizeT,
         typename GraphT,
         typename LambdaT>
sycl::event advance(const GraphT& graph, frontier<FrontierSizeT>& in, frontier<FrontierSizeT>& out, LambdaT&& functor, size_t expected_size = 0) {
  sycl::queue& q = graph.get_queue();

  size_t num_nodes = graph.get_vertex_count();

  using element_t = FrontierSizeT;
  // using element_t = std::conditional_t<InFW == sygraph::frontier::frontier_view::vertex, typename GraphT::vertex_t, typename GraphT::edge_t>;

  auto in_dev_frontier = in.get_device_view();
  auto out_dev_frontier = out.get_device_view();
  auto graph_dev = graph.get_device_view(); // GRAPH device view

  size_t coarsening_factor = 256 /*CU size*/ / 16 /*Subgroup Size*/;

  sycl::event to_wait;
  sycl::range<1> local_range;
  size_t global_size;
  size_t bitmap_range = in_dev_frontier.get_element_bitsize();
  to_wait = in.compute_active_frontier();
  local_range = {bitmap_range * coarsening_factor};
  if (expected_size > 0) {
    global_size = {expected_size + (local_range[0] - (expected_size % local_range[0]))};
  } else {
    global_size = {local_range[0] * 128};
    // global_size = {local_range[0] * (sygraph::detail::device::getMaxComputeUints(q) * 8)};
  }

  sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

  Context<decltype(in_dev_frontier), decltype(out_dev_frontier)> context{num_nodes, in_dev_frontier, out_dev_frontier};
  using bitmap_kernel_t = BitmapKernel<element_t, decltype(context), decltype(graph_dev), LambdaT>;

  const uint32_t max_num_subgroups = 32;
  // const uint32_t max_num_subgroups = sygraph::detail::device::getMaxNumSubgroups(q);

  auto e = q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(to_wait);
    sycl::local_accessor<uint32_t, 1> n_edges_wg{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> n_edges_sg{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<element_t, 1> subgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{max_num_subgroups, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_ids{local_range, cgh};
    sycl::local_accessor<element_t, 1> workgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail{1, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_ids{local_range, cgh};
    sycl::local_accessor<element_t, 1> individual_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> individual_reduce_tail{1, cgh};


    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
      bitmap_kernel_t{context,
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
                      std::forward<LambdaT>(functor)});
  });
  return {e};
}

} // namespace oneapi::dal::backend::primitives