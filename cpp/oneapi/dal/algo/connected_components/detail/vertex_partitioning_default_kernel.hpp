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

#pragma once

#include "oneapi/dal/algo/connected_components/common.hpp"
#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::connected_components::detail {

using namespace dal::preview::detail;

template <typename Method, typename Task, typename Allocator, typename Graph>
struct vertex_partitioning_kernel_cpu {
    inline vertex_partitioning_result<Task> operator()(const dal::detail::host_policy& ctx,
                                                       const detail::descriptor_base<Task>& desc,
                                                       const Allocator& alloc,
                                                       const Graph& g) const;
};

template <typename Float, typename Task, typename Topology, typename... Param>
struct afforest {
    vertex_partitioning_result<Task> operator()(const dal::detail::host_policy& ctx,
                                                const detail::descriptor_base<Task>& desc,
                                                const Topology& t,
                                                byte_alloc_iface* alloc) const;
};

template <typename Float>
struct afforest<Float, task::vertex_partitioning, dal::preview::detail::topology<std::int32_t>> {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::vertex_partitioning>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        byte_alloc_iface* alloc) const;
};

template <typename Allocator, typename Graph>
struct vertex_partitioning_kernel_cpu<method::afforest,
                                      task::vertex_partitioning,
                                      Allocator,
                                      Graph> {
    inline vertex_partitioning_result<task::vertex_partitioning> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::vertex_partitioning>& desc,
        const Allocator& alloc,
        const Graph& g) const {
        using topology_type = typename graph_traits<Graph>::impl_type::topology_type;
        auto topology_builder = dal::preview::detail::csr_topology_builder<Graph>();
        const auto& t = topology_builder(g);
        alloc_connector<Allocator> alloc_con(alloc);
        const auto vertex_count = t.get_vertex_count();
        if (vertex_count == 0) {
            return vertex_partitioning_result<task::vertex_partitioning>();
        }
        return afforest<float, task::vertex_partitioning, topology_type>{}(ctx,
                                                                           desc,
                                                                           t,
                                                                           &alloc_con);
    }
};

} // namespace oneapi::dal::preview::connected_components::detail
