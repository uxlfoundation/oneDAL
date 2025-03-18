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

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_topology_builder.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

using namespace dal::preview::detail;

template <typename Method, typename Task, typename Allocator, typename Graph>
struct traverse_kernel_cpu {
    inline traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                            const detail::descriptor_base<Task>& desc,
                                            const Allocator& alloc,
                                            const Graph& g) const;
};

template <typename Float, typename Task, typename Topology, typename EdgeValue, typename... Param>
struct delta_stepping {
    traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                     const detail::descriptor_base<Task>& desc,
                                     const Topology& t,
                                     const EdgeValue* vals,
                                     byte_alloc_iface* alloc) const;
};

template <typename Float, typename EdgeValue>
struct delta_stepping<Float,
                      task::one_to_all,
                      dal::preview::detail::topology<std::int32_t>,
                      EdgeValue> {
    traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc) const;
};

template <typename Float, typename Task, typename Topology, typename EdgeValue, typename... Param>
struct delta_stepping_with_pred {
    traverse_result<Task> operator()(const dal::detail::host_policy& ctx,
                                     const detail::descriptor_base<Task>& desc,
                                     const Topology& t,
                                     const EdgeValue* vals,
                                     byte_alloc_iface* alloc) const;
};

template <typename Float, typename EdgeValue>
struct delta_stepping_with_pred<Float,
                                task::one_to_all,
                                dal::preview::detail::topology<std::int32_t>,
                                EdgeValue> {
    traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc) const;
};

template <typename Allocator, typename Graph>
struct traverse_kernel_cpu<method::delta_stepping, task::one_to_all, Allocator, Graph> {
    inline traverse_result<task::one_to_all> operator()(
        const dal::detail::host_policy& ctx,
        const detail::descriptor_base<task::one_to_all>& desc,
        const Allocator& alloc,
        const Graph& g) const {
        using topology_type = typename graph_traits<Graph>::impl_type::topology_type;
        using value_type = edge_user_value_type<Graph>;
        auto topology_builder = dal::preview::detail::csr_topology_builder<Graph>();
        const auto& t = topology_builder(g);
        const auto vals = dal::detail::get_impl(g).get_edge_values().get_data();
        alloc_connector<Allocator> alloc_con(alloc);
        if (desc.get_optional_results() & optional_results::predecessors) {
            return delta_stepping_with_pred<float, task::one_to_all, topology_type, value_type>{}(
                ctx,
                desc,
                t,
                vals,
                &alloc_con);
        }
        else {
            return delta_stepping<float, task::one_to_all, topology_type, value_type>{}(ctx,
                                                                                        desc,
                                                                                        t,
                                                                                        vals,
                                                                                        &alloc_con);
        }
    }
};

} // namespace oneapi::dal::preview::shortest_paths::detail
