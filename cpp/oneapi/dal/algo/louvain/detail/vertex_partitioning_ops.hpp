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

#include "oneapi/dal/algo/louvain/common.hpp"
#include "oneapi/dal/algo/louvain/detail/select_kernel.hpp"
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::louvain::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct vertex_partitioning_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    vertex_partitioning_result<task_t> operator()(
        const Policy &policy,
        const Descriptor &descriptor,
        vertex_partitioning_input<Graph, task_t> &input) const {
        static auto impl = get_backend<Policy, Descriptor>(descriptor, input.get_graph());
        return (*impl)(policy, descriptor, input.get_graph(), input.get_initial_partition());
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

    void check_preconditions(const Descriptor &desc, input_t &input) const {
        using msg = dal::detail::error_messages;
        if (desc.get_accuracy_threshold() < 0) {
            throw invalid_argument(msg::accuracy_threshold_lt_zero());
        }
        if (desc.get_resolution() < 0) {
            throw invalid_argument(msg::negative_resolution());
        }
        if (desc.get_max_iteration_count() < 0) {
            throw invalid_argument(msg::max_iteration_count_lt_zero());
        }
        if (input.get_initial_partition().has_data()) {
            const std::int64_t vertex_count =
                dal::detail::get_impl(input.get_graph()).get_topology()._vertex_count;
            if (input.get_initial_partition().get_row_count() != vertex_count) {
                throw invalid_argument(msg::input_initial_partition_table_rc_neq_vertex_count());
            }
            if (input.get_initial_partition().get_column_count() != 1) {
                throw invalid_argument(
                    msg::input_initial_partition_table_has_wrong_cc_expect_one());
            }
            auto init_partition_arr =
                oneapi::dal::row_accessor<const std::int32_t>(input.get_initial_partition()).pull();
            const auto init_partition_data = init_partition_arr.get_data();
            std::int32_t min_label = std::numeric_limits<std::int32_t>::max();
            std::int32_t max_label = std::numeric_limits<std::int32_t>::min();
            for (std::int64_t v = 0; v < vertex_count; ++v) {
                min_label = std::min(min_label, init_partition_data[v]);
                max_label = std::max(max_label, init_partition_data[v]);
            }
            if (min_label < 0) {
                throw invalid_argument(msg::negative_initial_partition_label());
            }
            if (max_label >= vertex_count) {
                throw invalid_argument(msg::initial_partition_label_gte_vertex_count());
            }
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        check_preconditions(desc, input);
        return vertex_partitioning_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::louvain::detail
