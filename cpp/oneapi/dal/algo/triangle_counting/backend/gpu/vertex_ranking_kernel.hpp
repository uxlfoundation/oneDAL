#pragma once

#include <cstdint>

#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Index = std::int32_t>
struct csr_topology_gpu_view {
    const std::int64_t* rows = nullptr;
    const Index* cols = nullptr;
    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;
};

/// Creates a GPU view from a device_csr_topology (zero-copy, just extracts pointers).
template <typename Index = std::int32_t>
csr_topology_gpu_view<Index> make_gpu_view(
        const dal::preview::detail::device_csr_topology<Index>& device_topo) {
    csr_topology_gpu_view<Index> view;
    view.rows = device_topo.get_rows();
    view.cols = device_topo.get_cols();
    view.vertex_count = device_topo.get_vertex_count();
    view.edge_count = device_topo.get_edge_count();
    return view;
}

template <typename Float, typename Task, typename Index = std::int32_t>
vertex_ranking_result<Task> run_vertex_ranking_gpu(const dal::backend::context_gpu& ctx,
                                                   const detail::descriptor_base<Task>& desc,
                                                   const csr_topology_gpu_view<Index>& topology);

} // namespace oneapi::dal::preview::triangle_counting::backend
