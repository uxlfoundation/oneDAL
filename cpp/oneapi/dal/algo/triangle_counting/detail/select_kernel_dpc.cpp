/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/vertex_ranking_ops.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/gpu/triangle_counting.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/gpu/vertex_ranking_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Descriptor, typename Topology>
vertex_ranking_result<typename Descriptor::task_t>
backend_default<dal::detail::data_parallel_policy, Descriptor, Topology>::operator()(
    const dal::detail::data_parallel_policy &ctx,
    const Descriptor &descriptor,
    const Topology &t) {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;

    return dal::backend::dispatch_by_device(
        ctx,
        [&]() {
            // CPU path: delegate to host policy kernel
            return vertex_ranking_kernel_cpu<method_t, task_t, allocator_t, Topology>()(
                dal::detail::host_policy::get_default(),
                descriptor,
                descriptor.get_allocator(),
                t);
        },
        [&]() {
            // GPU path: delegate to GPU kernel
            dal::backend::context_gpu gpu_ctx{ ctx };
            return backend::vertex_ranking_kernel_gpu<float_t, task_t, Topology>()(gpu_ctx,
                                                                                   descriptor,
                                                                                   t);
        });
}

/// Specialization for device_csr_topology: GPU-only path (no CPU fallback).
/// Handled directly in vertex_ranking_ops_dispatcher.

template <typename Descriptor, typename Index>
vertex_ranking_result<typename Descriptor::task_t>
vertex_ranking_ops_dispatcher<dal::detail::data_parallel_policy,
                              Descriptor,
                              dal::preview::detail::device_csr_topology<Index>>::
operator()(const dal::detail::data_parallel_policy &policy,
           const Descriptor &descriptor,
           input_t &input) const {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;

    const auto &device_topo = input.get_graph();
    if (device_topo.get_vertex_count() == 0) {
        return vertex_ranking_result<task_t>();
    }
    dal::backend::context_gpu gpu_ctx{ policy };
    const auto gpu_view = backend::make_gpu_view(device_topo);
    return backend::run_vertex_ranking_gpu<float_t, task_t, Index>(gpu_ctx, descriptor, gpu_view);
}

// Explicit instantiations for supported descriptor/topology combinations
using default_descriptor_local =
    descriptor<float, method::ordered_count, task::local, std::allocator<char>>;

using default_descriptor_global =
    descriptor<float, method::ordered_count, task::global, std::allocator<char>>;

using default_descriptor_local_and_global =
    descriptor<float, method::ordered_count, task::local_and_global, std::allocator<char>>;

using default_topology = dal::preview::detail::topology<std::int32_t>;
using device_topology = dal::preview::detail::device_csr_topology<std::int32_t>;

template struct ONEDAL_EXPORT
    backend_default<dal::detail::data_parallel_policy, default_descriptor_local, default_topology>;
template struct ONEDAL_EXPORT
    backend_default<dal::detail::data_parallel_policy, default_descriptor_global, default_topology>;
template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              default_descriptor_local_and_global,
                                              default_topology>;

// Explicit instantiations for device_csr_topology dispatcher
template struct vertex_ranking_ops_dispatcher<dal::detail::data_parallel_policy,
                                              default_descriptor_local,
                                              device_topology>;
template struct vertex_ranking_ops_dispatcher<dal::detail::data_parallel_policy,
                                              default_descriptor_global,
                                              device_topology>;
template struct vertex_ranking_ops_dispatcher<dal::detail::data_parallel_policy,
                                              default_descriptor_local_and_global,
                                              device_topology>;

} // namespace oneapi::dal::preview::triangle_counting::detail
