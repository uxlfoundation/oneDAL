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

#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/rng/rng.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;
using namespace oneapi::dal::backend::primitives;

template <typename IndexType, typename EdgeValue>
struct louvain_data {
    using value_type = EdgeValue;
    using vertex_type = std::int32_t;
    using vertex_size_type = std::int64_t;

    using value_allocator_type = inner_alloc<value_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;
    using vertex_size_allocator_type = inner_alloc<vertex_size_type>;

    louvain_data() = delete;
    louvain_data(std::int64_t vertex_count,
                 std::int64_t edge_count,
                 value_allocator_type& value_allocator,
                 vertex_allocator_type& vertex_allocator,
                 vertex_size_allocator_type& vertex_size_allocator)
            : m(0),
              vertex_count(vertex_count),
              edge_count(edge_count),
              value_allocator(value_allocator),
              vertex_allocator(vertex_allocator),
              vertex_size_allocator(vertex_size_allocator) {
        k = allocate(value_allocator, vertex_count);
        tot = allocate(value_allocator, vertex_count);
        k_vertex_to = allocate(value_allocator, vertex_count);
        neighboring_communities = allocate(vertex_allocator, vertex_count);
        random_order = allocate(vertex_allocator, vertex_count);
        empty_community = allocate(vertex_allocator, vertex_count);
        community_size = allocate(vertex_size_allocator, vertex_count);

        k_c = allocate(value_allocator, vertex_count);
        local_self_loops = allocate(value_allocator, vertex_count);

        weights = allocate(value_allocator, vertex_count);
        c_self_loops = allocate(value_allocator, vertex_count);
        c_neighbors = allocate(vertex_allocator, vertex_count);
        c_rows = allocate(vertex_allocator, vertex_count + 1);
        c2v = allocate(vertex_allocator, vertex_count);
        community_index = allocate(vertex_size_allocator, vertex_count + 1);
        prefix_sum = allocate(vertex_size_allocator, vertex_count + 1);

        c_vals = allocate(value_allocator, edge_count * 2);
        c_cols = allocate(vertex_allocator, edge_count * 2);
        index = allocate(vertex_allocator, vertex_count);
    }
    ~louvain_data() {
        deallocate(value_allocator, k, vertex_count);
        deallocate(value_allocator, tot, vertex_count);
        deallocate(value_allocator, k_vertex_to, vertex_count);
        deallocate(vertex_allocator, neighboring_communities, vertex_count);
        deallocate(vertex_allocator, random_order, vertex_count);
        deallocate(vertex_allocator, empty_community, vertex_count);
        deallocate(vertex_size_allocator, community_size, vertex_count);

        deallocate(value_allocator, k_c, vertex_count);
        deallocate(value_allocator, local_self_loops, vertex_count);

        deallocate(value_allocator, weights, vertex_count);
        deallocate(value_allocator, c_self_loops, vertex_count);
        deallocate(vertex_allocator, c_neighbors, vertex_count);
        deallocate(vertex_allocator, c_rows, vertex_count + 1);
        deallocate(vertex_allocator, c2v, vertex_count);
        deallocate(vertex_size_allocator, community_index, vertex_count + 1);
        deallocate(vertex_size_allocator, prefix_sum, vertex_count + 1);

        deallocate(value_allocator, c_vals, edge_count * 2);
        deallocate(vertex_allocator, c_cols, edge_count * 2);
        deallocate(vertex_allocator, index, vertex_count);
    }

    // Sum of the weights of the edges attached to nodes
    value_type* k;
    // Sum of the weights of the links incident to vertices in community
    value_type* tot;
    // Sum of weights from current vertex to communies
    value_type* k_vertex_to;
    vertex_type* neighboring_communities;
    vertex_type* random_order;
    vertex_type* empty_community;
    vertex_size_type* community_size;

    value_type* k_c;
    value_type* local_self_loops;

    value_type* weights;
    value_type* c_self_loops;
    vertex_type* c_neighbors;
    vertex_type* c_rows;
    vertex_type* c2v;
    vertex_size_type* community_index;
    vertex_size_type* prefix_sum;

    value_type* c_vals;
    vertex_type* c_cols;
    vertex_type* index;

    // Total link weight in the network
    value_type m;

    daal_engine<engine_list::mt2203> eng;
    rng<std::int32_t> rn_gen;

    const std::int64_t vertex_count;
    const std::int64_t edge_count;

    value_allocator_type& value_allocator;
    vertex_allocator_type& vertex_allocator;
    vertex_size_allocator_type& vertex_size_allocator;
};

} // namespace oneapi::dal::preview::louvain::backend
