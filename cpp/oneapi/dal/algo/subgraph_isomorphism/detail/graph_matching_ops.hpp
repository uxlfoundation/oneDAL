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

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/select_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct graph_matching_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    graph_matching_result<task_t> operator()(const Policy &policy,
                                             const Descriptor &descriptor,
                                             graph_matching_input<Graph, task_t> &input) const {
        auto topology_builder = dal::preview::detail::csr_topology_builder<Graph>();
        const auto &csr_target_topology = topology_builder(input.get_target_graph());
        const auto &csr_pattern_topology = topology_builder(input.get_pattern_graph());
        const auto &vv_t = dal::detail::get_impl(input.get_target_graph()).get_vertex_values();
        const auto &ev_t = dal::detail::get_impl(input.get_target_graph()).get_edge_values();
        const auto &vv_p = dal::detail::get_impl(input.get_pattern_graph()).get_vertex_values();
        const auto &ev_p = dal::detail::get_impl(input.get_pattern_graph()).get_edge_values();

        static auto impl =
            get_backend<Policy>(descriptor, csr_target_topology, csr_pattern_topology, vv_t, ev_t);

        return (*impl)(policy,
                       descriptor,
                       csr_target_topology,
                       csr_pattern_topology,
                       vv_t,
                       ev_t,
                       vv_p,
                       ev_p);
    }
};

template <typename Descriptor, typename Graph>
struct graph_matching_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;
    using graph_t = Graph;
    using input_t = graph_matching_input<graph_t, task_t>;
    using result_t = graph_matching_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor &desc, input_t &input) const {
        using msg = dal::detail::error_messages;

        if (dal::preview::get_vertex_count(input.get_target_graph()) == 0) {
            throw invalid_argument(msg::empty_target_graph());
        }
        if (dal::preview::get_vertex_count(input.get_pattern_graph()) == 0) {
            throw invalid_argument(msg::empty_pattern_graph());
        }
        if (dal::preview::get_vertex_count(input.get_target_graph()) <
            dal::preview::get_vertex_count(input.get_pattern_graph())) {
            throw invalid_argument(msg::target_graph_is_smaller_than_pattern_graph());
        }
        if (desc.get_max_match_count() < 0) {
            throw invalid_argument(msg::max_match_count_lt_zero());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        check_preconditions(desc, input);
        return graph_matching_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
