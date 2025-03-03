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

#pragma once

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/detail/select_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct vertex_ranking_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    vertex_ranking_result<task_t> operator()(const Policy &policy,
                                             const Descriptor &descriptor,
                                             vertex_ranking_input<Graph, task_t> &input) const {
        auto topology_builder = dal::preview::detail::csr_topology_builder<Graph>();

        const auto &t = topology_builder(input.get_graph());

        static auto impl = get_backend<Policy, Descriptor>(descriptor, t);

        return (*impl)(policy, descriptor, t);
    }
};

template <typename Descriptor, typename Graph>
struct vertex_ranking_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;
    using graph_t = Graph;
    using input_t = vertex_ranking_input<graph_t, task_t>;
    using result_t = vertex_ranking_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        return vertex_ranking_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::triangle_counting::detail
