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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/detail/select_kernel.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_topology_builder.hpp"

namespace oneapi::dal::preview::jaccard::detail {

template <typename Policy, typename Descriptor, typename Graph>
struct vertex_similarity_ops_dispatcher {
    using task_t = typename Descriptor::task_t;
    vertex_similarity_result<task_t> operator()(
        const Policy &policy,
        const Descriptor &descriptor,
        vertex_similarity_input<Graph, task_t> &input) const {
        const auto &t = dal::preview::detail::csr_topology_builder<Graph>()(input.get_graph());

        static auto impl = get_backend<Policy, Descriptor>(descriptor, t);
        return (*impl)(policy, descriptor, t, input.get_caching_builder());
    }
};

template <typename Descriptor, typename Graph>
struct vertex_similarity_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using graph_t = Graph;
    using input_t = vertex_similarity_input<graph_t, task_t>;
    using result_t = vertex_similarity_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor &param, input_t &input) const {
        using msg = dal::detail::error_messages;

        const std::int64_t row_begin = param.get_row_range_begin();
        const std::int64_t row_end = param.get_row_range_end();
        const std::int64_t column_begin = param.get_column_range_begin();
        const std::int64_t column_end = param.get_column_range_end();
        if (row_begin < 0 || column_begin < 0) {
            throw invalid_argument(msg::negative_interval());
        }
        if (row_begin > row_end) {
            throw invalid_argument(msg::row_begin_gt_row_end());
        }
        if (column_begin > column_end) {
            throw invalid_argument(msg::column_begin_gt_column_end());
        }
        const std::int64_t vertex_count =
            dal::detail::get_impl(input.get_graph()).get_topology()._vertex_count;
        // Safe conversion as ranges were checked
        if (row_end > vertex_count || column_end > vertex_count) {
            throw out_of_range(msg::interval_gt_vertex_count());
        }
        if (row_end >= dal::detail::limits<std::int32_t>::max() ||
            column_end >= dal::detail::limits<std::int32_t>::max()) {
            throw invalid_argument(msg::range_idx_gt_max_int32());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy, const Descriptor &desc, input_t &input) const {
        check_preconditions(desc, input);
        return vertex_similarity_ops_dispatcher<Policy, Descriptor, Graph>()(policy, desc, input);
    }
};

} // namespace oneapi::dal::preview::jaccard::detail
