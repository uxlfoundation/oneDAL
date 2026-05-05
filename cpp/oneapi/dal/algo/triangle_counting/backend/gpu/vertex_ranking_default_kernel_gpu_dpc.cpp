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

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/triangle_counting/backend/gpu/triangle_counting.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/gpu/vertex_ranking_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

namespace detail_gpu {

/// Counts triangles for each vertex in a CSR graph on the GPU using a
/// binary-search intersection approach.
///
/// @tparam Index  The index type used for vertex/edge indexing
/// @param[in]  queue         SYCL queue for device execution
/// @param[in]  rows          CSR row-offset array (vertex_count + 1 entries)
/// @param[in]  cols          CSR column-index array (2 * edge_count entries)
/// @param[in]  vertex_count  Number of vertices in the graph
/// @param[in]  edge_count    Number of undirected edges
/// @return     A USM-allocated array of per-vertex triangle counts
template <typename Index>
std::int64_t* count_triangles_gpu(sycl::queue& queue,
                                  const std::int64_t* rows,
                                  const Index* cols,
                                  std::int64_t vertex_count,
                                  std::int64_t edge_count) {
    auto* local_triangles =
        sycl::malloc_shared<std::int64_t>(vertex_count, queue);
    queue.memset(local_triangles, 0, vertex_count * sizeof(std::int64_t)).wait_and_throw();

    // Parallel kernel: each work-item processes one vertex
    queue.submit([&](sycl::handler& cgh) {
        const std::int64_t* d_rows = rows;
        const Index* d_cols = cols;
        std::int64_t* d_triangles = local_triangles;
        const std::int64_t vc = vertex_count;

        cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
            const std::int64_t u = idx[0];
            const std::int64_t u_start = d_rows[u];
            const std::int64_t u_end = d_rows[u + 1];

            std::int64_t count = 0;

            // Iterate over each neighbor v of u where v > u
            for (std::int64_t i = u_start; i < u_end; ++i) {
                const std::int64_t v = d_cols[i];
                if (v <= u) continue;

                const std::int64_t v_start = d_rows[v];
                const std::int64_t v_end = d_rows[v + 1];

                // Merge-based intersection of neighbors of u and v
                std::int64_t ui = u_start;
                std::int64_t vi = v_start;
                while (ui < u_end && vi < v_end) {
                    const std::int64_t nu = d_cols[ui];
                    const std::int64_t nv = d_cols[vi];
                    if (nu == nv) {
                        // Only count if common neighbor w > v to avoid
                        // triple-counting the same triangle
                        if (nu > v) {
                            // Found a triangle (u, v, nu) with u < v < nu
                            count++;
                            // Atomically increment counts for v and the common neighbor
                            sycl::atomic_ref<std::int64_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(
                                d_triangles[v])
                                .fetch_add(1);
                            sycl::atomic_ref<std::int64_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(
                                d_triangles[nu])
                                .fetch_add(1);
                        }
                        ++ui;
                        ++vi;
                    }
                    else if (nu < nv) {
                        ++ui;
                    }
                    else {
                        ++vi;
                    }
                }
            }
            // Add the count for vertex u
            sycl::atomic_ref<std::int64_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(d_triangles[u])
                .fetch_add(count);
        });
    }).wait_and_throw();

    return local_triangles;
}

/// Sums per-vertex triangle counts and divides by 3 to get the global count.
///
/// @param[in]  queue           SYCL queue for device execution
/// @param[in]  local_triangles Per-vertex triangle count array
/// @param[in]  vertex_count    Number of vertices
/// @return     The global (total) triangle count
std::int64_t sum_triangles_gpu(sycl::queue& queue,
                               const std::int64_t* local_triangles,
                               std::int64_t vertex_count) {
    auto* sum_buf = sycl::malloc_shared<std::int64_t>(1, queue);
    sum_buf[0] = 0;

    queue.submit([&](sycl::handler& cgh) {
        const std::int64_t* d_triangles = local_triangles;
        std::int64_t* d_sum = sum_buf;
        const std::int64_t vc = vertex_count;

        cgh.parallel_for(sycl::range<1>(vc), [=](sycl::id<1> idx) {
            sycl::atomic_ref<std::int64_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(d_sum[0])
                .fetch_add(d_triangles[idx[0]]);
        });
    }).wait_and_throw();

    // Each triangle is counted 3 times (once per vertex)
    const std::int64_t result = sum_buf[0] / 3;
    sycl::free(sum_buf, queue);
    return result;
}

} // namespace detail_gpu

template <typename Float, typename Task, typename Index>
vertex_ranking_result<Task> run_vertex_ranking_gpu(const dal::backend::context_gpu& ctx,
                                                   const detail::descriptor_base<Task>& desc,
                                                   const csr_topology_gpu_view<Index>& topology) {
    auto& queue = ctx.get_queue();

    const auto vertex_count = topology.vertex_count;
    const auto edge_count = topology.edge_count;

    if (vertex_count == 0) {
        return vertex_ranking_result<Task>();
    }

    auto* local_triangles =
        detail_gpu::count_triangles_gpu(queue, topology.rows, topology.cols,
                                        vertex_count, edge_count);

    vertex_ranking_result<Task> result;

    if constexpr (std::is_same_v<Task, task::local> ||
                  std::is_same_v<Task, task::local_and_global>) {
        auto arr = array<std::int64_t>(queue, local_triangles, vertex_count,
                                       [queue](std::int64_t* ptr) mutable {
                                           sycl::free(ptr, queue);
                                       });
        result.set_ranks(dal::detail::homogen_table_builder{}
                             .reset(arr, vertex_count, 1)
                             .build());
    }

    if constexpr (std::is_same_v<Task, task::global> ||
                  std::is_same_v<Task, task::local_and_global>) {
        std::int64_t global_count =
            detail_gpu::sum_triangles_gpu(queue, local_triangles, vertex_count);
        result.set_global_rank(global_count);
    }

    // Free local_triangles only if not wrapped into the result table
    if constexpr (std::is_same_v<Task, task::global>) {
        sycl::free(local_triangles, queue);
    }

    return result;
}

template <typename Float, typename Task, typename Topology>
vertex_ranking_result<Task> vertex_ranking_kernel_gpu<Float, Task, Topology>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& desc,
    const Topology& topology) const {
    auto& queue = ctx.get_queue();
    const auto vertex_count = topology.get_vertex_count();

    if (vertex_count == 0) {
        return vertex_ranking_result<Task>();
    }

    using namespace dal::preview::detail;

    const auto* rows_ptr = topology._rows.get_data();
    const auto alloc_type = sycl::get_pointer_type(rows_ptr, queue.get_context());

    if (alloc_type == sycl::usm::alloc::device || alloc_type == sycl::usm::alloc::shared) {
        csr_topology_gpu_view<std::int32_t> gpu_view;
        gpu_view.rows = rows_ptr;
        gpu_view.cols = topology._cols.get_data();
        gpu_view.vertex_count = vertex_count;
        gpu_view.edge_count = topology.get_edge_count();
        return run_vertex_ranking_gpu<Float, Task, std::int32_t>(ctx, desc, gpu_view);
    }

    auto device_topo = topology_to_device<std::int32_t>(queue, topology);
    const auto gpu_view = make_gpu_view(device_topo);
    return run_vertex_ranking_gpu<Float, Task, std::int32_t>(ctx, desc, gpu_view);
}

// Explicit template instantiations
template struct vertex_ranking_kernel_gpu<float,
                                          task::local,
                                          dal::preview::detail::topology<std::int32_t>>;
template struct vertex_ranking_kernel_gpu<float,
                                          task::global,
                                          dal::preview::detail::topology<std::int32_t>>;
template struct vertex_ranking_kernel_gpu<float,
                                          task::local_and_global,
                                          dal::preview::detail::topology<std::int32_t>>;

template vertex_ranking_result<task::local>
run_vertex_ranking_gpu<float, task::local, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::local>&,
    const csr_topology_gpu_view<std::int32_t>&);

template vertex_ranking_result<task::global>
run_vertex_ranking_gpu<float, task::global, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::global>&,
    const csr_topology_gpu_view<std::int32_t>&);

template vertex_ranking_result<task::local_and_global>
run_vertex_ranking_gpu<float, task::local_and_global, std::int32_t>(
    const dal::backend::context_gpu&,
    const detail::descriptor_base<task::local_and_global>&,
    const csr_topology_gpu_view<std::int32_t>&);

} // namespace oneapi::dal::preview::triangle_counting::backend
