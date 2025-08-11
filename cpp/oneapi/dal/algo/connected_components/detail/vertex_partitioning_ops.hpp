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
#include "oneapi/dal/algo/connected_components/detail/select_kernel.hpp"
#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::connected_components::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct vertex_partitioning_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    vertex_partitioning_result<task_t> operator()(
        const Policy &policy,
        const Descriptor &descriptor,
        vertex_partitioning_input<Graph, task_t> &input) const {
        static auto impl = get_backend<Policy, Descriptor>(descriptor, input.get_graph());
        return (*impl)(policy, descriptor, input.get_graph());
    }
};

template <typename Descriptor, typename Graph>
struct vertex_partitioning_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;
    using graph_t = Graph;
    using input_t = vertex_partitioning_input<graph_t, task_t>;
    using result_t = vertex_partitioning_result<task_t>;

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        return vertex_partitioning_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::connected_components::detail
