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

#include "oneapi/dal/algo/shortest_paths/detail/select_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/backend/gpu/shortest_paths.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

template <typename Descriptor, typename Graph>
traverse_result<typename Descriptor::task_t>
backend_default<dal::detail::data_parallel_policy, Descriptor, Graph>::operator()(
    const dal::detail::data_parallel_policy& ctx,
    const Descriptor& descriptor,
    const Graph& g) {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;
    using value_type = edge_user_value_type<Graph>;

    return dal::backend::dispatch_by_device(
        ctx,
        [&]() {
            return traverse_kernel_cpu<method_t, task_t, allocator_t, Graph>()(
                dal::detail::host_policy::get_default(),
                descriptor,
                descriptor.get_allocator(),
                g);
        },
        [&]() {
            using topology_type =
                typename graph_traits<Graph>::impl_type::topology_type;
            auto topology_builder = dal::preview::detail::csr_topology_builder<Graph>();
            const auto& t = topology_builder(g);
            const auto vals = dal::detail::get_impl(g).get_edge_values().get_data();
            dal::backend::context_gpu gpu_ctx{ ctx };
            return backend::traverse_kernel_gpu<float_t, task_t, topology_type, value_type>()(
                gpu_ctx,
                descriptor,
                t,
                vals);
        });
}

// Explicit instantiations for int32_t edge weights
using descriptor_int32 = descriptor<
    float,
    method::delta_stepping,
    task::one_to_all,
    std::allocator<char>>;

using graph_int32 = dal::preview::directed_adjacency_vector_graph<
    std::int32_t, std::int32_t, dal::preview::empty_value, int, std::allocator<char>>;

template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              descriptor_int32,
                                              graph_int32>;

// Explicit instantiations for double edge weights
using descriptor_double = descriptor<
    float,
    method::delta_stepping,
    task::one_to_all,
    std::allocator<char>>;

using graph_double = dal::preview::directed_adjacency_vector_graph<
    std::int32_t, double, dal::preview::empty_value, int, std::allocator<char>>;

template struct ONEDAL_EXPORT backend_default<dal::detail::data_parallel_policy,
                                              descriptor_double,
                                              graph_double>;

} // namespace oneapi::dal::preview::shortest_paths::detail
