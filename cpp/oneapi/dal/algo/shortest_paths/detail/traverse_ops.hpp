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
#include "oneapi/dal/algo/shortest_paths/detail/select_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::shortest_paths::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct traverse_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    traverse_result<task_t> operator()(const Policy &policy,
                                       const Descriptor &descriptor,
                                       traverse_input<Graph, task_t> &input) const {
        static auto impl = get_backend<Policy, Descriptor>(descriptor, input.get_graph());
        return (*impl)(policy, descriptor, input.get_graph());
    }
};

template <typename Descriptor, typename Graph>
struct traverse_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using allocator_t = typename Descriptor::allocator_t;
    using graph_t = Graph;
    using input_t = traverse_input<graph_t, task_t>;
    using result_t = traverse_result<task_t>;

    template <typename T = task_t,
              typename M = method_t,
              typename = enable_if_delta_stepping_single_source_t<T, M>>
    void check_preconditions(const Descriptor &desc, input_t &input) const {
        using msg = dal::detail::error_messages;
        if (desc.get_source() < 0) {
            throw invalid_argument(msg::negative_source());
        }
        const std::int64_t vertex_count =
            dal::detail::get_impl(input.get_graph()).get_topology()._vertex_count;
        if (desc.get_source() >= vertex_count) {
            throw invalid_argument(msg::source_gte_vertex_count());
        }
        if (desc.get_delta() < 0) {
            throw invalid_argument(msg::negative_delta());
        }
        if (!(desc.get_optional_results() &
              (optional_results::predecessors | optional_results::distances))) {
            throw invalid_argument(msg::nothing_to_compute());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        check_preconditions(desc, input);
        return traverse_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::shortest_paths::detail
