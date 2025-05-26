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
#include "oneapi/dal/algo/louvain/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/algo/louvain/backend/cpu/louvain_data.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;
using namespace oneapi::dal::backend::primitives;

template <typename IndexType>
inline void singleton_partition(IndexType* labels, std::int64_t vertex_count) {
    for (IndexType v = 0; v < vertex_count; v++) {
        labels[v] = v;
    }
}

template <typename IndexType>
inline std::int64_t reindex_communities(IndexType* communities,
                                        std::int64_t* community_size,
                                        std::int64_t vertex_count,
                                        IndexType* index) {
    for (std::int64_t v = 0; v < vertex_count; v++) {
        index[v] = -1;
        community_size[v] = 0;
    }
    IndexType count = 0;
    for (std::int64_t v = 0; v < vertex_count; v++) {
        if (index[communities[v]] == -1) {
            index[communities[v]] = count++;
        }
        communities[v] = index[communities[v]];
        community_size[communities[v]]++;
    }
    return count;
}

template <typename IndexType, typename EdgeValue>
inline void compress_graph(dal::preview::detail::topology<IndexType>& t,
                           EdgeValue* vals,
                           EdgeValue* self_loops,
                           const std::int64_t community_count,
                           const IndexType* partition,
                           louvain_data<IndexType, EdgeValue>& ld) {
    ld.c_rows[0] = 0;
    for (std::int64_t c = 0; c < community_count; c++) {
        ld.c_self_loops[c] = 0;
        ld.weights[c] = 0;
    }
    ld.community_index[0] = ld.prefix_sum[0] = 0;
    for (std::int64_t c = 1; c <= community_count; c++) {
        ld.community_index[c] = ld.prefix_sum[c] = ld.prefix_sum[c - 1] + ld.community_size[c - 1];
    }
    for (IndexType v = 0; v < t._vertex_count; v++) {
        const IndexType c = partition[v];
        const std::int64_t index = ld.community_index[c]++;
        ld.c2v[index] = v;
    }

    std::int64_t neighbor_count = 0;
    for (IndexType c = 0; c < community_count; c++) {
        for (std::int64_t v_index = ld.prefix_sum[c]; v_index < ld.prefix_sum[c + 1]; v_index++) {
            const IndexType v = ld.c2v[v_index];
            ld.c_self_loops[c] += self_loops[v];
            for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
                const IndexType v_to = t._cols_ptr[index];
                const IndexType c_to = partition[v_to];
                const EdgeValue v_w = vals[index];
                if (c == c_to) {
                    if (v < v_to) {
                        ld.c_self_loops[c] += v_w;
                    }
                }
                else {
                    if (ld.weights[c_to] == 0) {
                        ld.c_neighbors[neighbor_count++] = c_to;
                    }
                    ld.weights[c_to] += v_w;
                }
            }
        }
        ld.c_rows[c + 1] = ld.c_rows[c] + neighbor_count;
        for (std::int64_t index = 0, c_index = ld.c_rows[c]; index < neighbor_count;
             index++, c_index++) {
            const IndexType c_neigh = ld.c_neighbors[index];
            ld.c_cols[c_index] = c_neigh;
            ld.c_vals[c_index] = ld.weights[c_neigh];
            ld.weights[c_neigh] = 0;
        }
        neighbor_count = 0;
    }

    std::int64_t* t_rows = t._rows.get_mutable_data();
    for (std::int64_t c = 0; c < community_count; c++) {
        self_loops[c] = ld.c_self_loops[c];
        t_rows[c + 1] = ld.c_rows[c + 1];
    }
    IndexType* t_cols = t._cols.get_mutable_data();
    for (std::int64_t index = 0; index < ld.c_rows[community_count]; index++) {
        t_cols[index] = ld.c_cols[index];
        vals[index] = ld.c_vals[index];
    }
}

template <typename Float, typename IndexType, typename EdgeValue>
inline Float init_step(const dal::preview::detail::topology<IndexType>& t,
                       const EdgeValue* vals,
                       const EdgeValue* self_loops,
                       const IndexType* labels,
                       const Float resolution,
                       louvain_data<IndexType, EdgeValue>& ld) {
    IndexType max_community_label = 0;
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        ld.community_size[labels[v]]++;
        max_community_label = std::max(max_community_label, labels[v]);
    }
    const std::int64_t community_count =
        dal::detail::integral_cast<std::int64_t>(max_community_label) + 1;
    for (std::int64_t c = 0; c < community_count; c++) {
        ld.k_c[c] = 0;
        ld.local_self_loops[c] = 0;
    }
    ld.m = 0;
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        const IndexType c = labels[v];
        ld.local_self_loops[c] += self_loops[v];
        ld.k_c[c] += self_loops[v] * 2;
        ld.k[v] += self_loops[v] * 2;
        ld.tot[c] += self_loops[v] * 2;
        ld.m += self_loops[v];
        for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
            const IndexType to = t._cols_ptr[index];
            const EdgeValue v_w = vals[index];
            const IndexType to_c = labels[to];
            ld.k_c[c] += v_w;
            ld.k[v] += v_w;
            ld.tot[c] += v_w;
            if (v < to) {
                ld.m += v_w;
                if (c == to_c) {
                    ld.local_self_loops[c] += v_w;
                }
            }
        }
    }
    ONEDAL_ASSERT(ld.m > 0);

    Float modularity = 0;
    if (ld.m > 0) {
        for (std::int64_t c = 0; c < community_count; c++) {
            modularity +=
                static_cast<Float>(0.5) / static_cast<Float>(ld.m) *
                (static_cast<Float>(ld.local_self_loops[c]) * static_cast<Float>(2) -
                 resolution * static_cast<Float>(ld.k_c[c]) * static_cast<Float>(ld.k_c[c]) /
                     (static_cast<Float>(2) * static_cast<Float>(ld.m)));
        }
    }
    return modularity;
}

template <typename Cpu, typename Float, typename IndexType, typename EdgeValue>
inline Float move_nodes(const dal::preview::detail::topology<IndexType>& t,
                        const EdgeValue* vals,
                        const EdgeValue* self_loops,
                        IndexType* n2c,
                        bool& changed,
                        const Float resolution,
                        const Float accuracy_threshold,
                        louvain_data<IndexType, EdgeValue>& ld) {
    for (std::int64_t v = 0; v < t._vertex_count; v++) {
        ld.k[v] = 0;
        ld.tot[v] = 0;
        ld.community_size[v] = 0;
        ld.k_vertex_to[v] = 0;
        ld.neighboring_communities[v] = 0;
    }

    // calc initial data
    Float modularity = init_step<Float>(t, vals, self_loops, n2c, resolution, ld);

    // interate over all vertices
    Float old_modularity = modularity;
    for (IndexType index = 0; index < t._vertex_count; index++) {
        ld.random_order[index] = index;
    }
    // random shuffle
    uniform<std::int32_t>(t._vertex_count, ld.index, ld.eng, 0, t._vertex_count);
    for (std::int64_t index = 0; index < t._vertex_count; ++index) {
        std::swap(ld.random_order[index], ld.random_order[ld.index[index]]);
    }
    std::int64_t empty_count = 0;
    do {
        old_modularity = modularity;
        for (std::int64_t order_index = 0; order_index < t._vertex_count; order_index++) {
            const IndexType v = ld.random_order[order_index];
            const IndexType c_old = n2c[v];

            // calculate sum of weights of edges between vertex and community to move into
            std::int64_t community_count = 0;
            for (std::int64_t index = t._rows_ptr[v]; index < t._rows_ptr[v + 1]; index++) {
                const IndexType to = t._cols_ptr[index];
                const IndexType c = n2c[to];
                const EdgeValue v_w = vals[index];
                if (ld.k_vertex_to[c] == 0) {
                    ld.neighboring_communities[community_count++] = c;
                }
                ld.k_vertex_to[c] += v_w;
            }

            // remove vertex from the current community
            const EdgeValue k_iold = ld.k_vertex_to[c_old];
            ld.tot[c_old] -= ld.k[v];
            Float delta_modularity =
                static_cast<Float>(k_iold) / static_cast<Float>(ld.m) -
                resolution * static_cast<Float>(ld.tot[c_old]) * static_cast<Float>(ld.k[v]) /
                    (static_cast<Float>(2) * static_cast<Float>(ld.m) * static_cast<Float>(ld.m));
            modularity -= delta_modularity;
            IndexType move_community = n2c[v];
            ld.community_size[c_old]--;
            if (!ld.community_size[c_old]) {
                ld.empty_community[empty_count++] = c_old;
            }
            // optionaly can be removed, but c_old community can be checked twice
            else if (empty_count) {
                ld.neighboring_communities[community_count++] = ld.empty_community[empty_count - 1];
            }

            // iterate over nodes
            for (std::int64_t index = 0; index < community_count; index++) {
                const IndexType c = ld.neighboring_communities[index];

                // try to move vertex to the community
                const EdgeValue k_ic = ld.k_vertex_to[c];
                const Float delta = static_cast<Float>(k_ic) / static_cast<Float>(ld.m) -
                                    resolution * static_cast<Float>(ld.tot[c]) *
                                        static_cast<Float>(ld.k[v]) /
                                        (static_cast<Float>(2) * static_cast<Float>(ld.m) *
                                         static_cast<Float>(ld.m));
                if (delta_modularity < delta) {
                    delta_modularity = delta;
                    move_community = c;
                }
                ld.k_vertex_to[c] = 0;
            }
            ld.k_vertex_to[c_old] = 0;

            // move vertex to the best community with the best modularity gain
            modularity += delta_modularity;
            ld.tot[move_community] += ld.k[v];
            n2c[v] = move_community;
            if (move_community != c_old) {
                changed = true;
            }
            ld.community_size[move_community]++;
            if (ld.community_size[move_community] == 1) {
                empty_count--;
            }
        }
    } while (modularity - old_modularity > accuracy_threshold);

    return modularity;
}

template <typename IndexType, typename CommunityVector, typename SizeVector>
inline void set_result_labels(CommunityVector& communities,
                              const SizeVector& vertex_size,
                              const IndexType* init_partition,
                              std::int64_t vertex_count,
                              IndexType* result_labels) {
    if (!communities.empty()) {
        // flat the communities from the next iteration
        for (std::int64_t iteration = communities.size() - 2; iteration >= 0; iteration--) {
            for (std::int64_t v = 0; v < vertex_size[iteration]; v++) {
                communities[iteration][v] = communities[iteration + 1][communities[iteration][v]];
            }
        }
        for (std::int64_t v = 0; v < vertex_count; v++) {
            result_labels[v] = communities[0][v];
        }
    }
    else if (init_partition == nullptr) {
        for (IndexType v = 0; v < vertex_count; v++) {
            result_labels[v] = v;
        }
    }
    else {
        for (std::int64_t v = 0; v < vertex_count; v++) {
            result_labels[v] = init_partition[v];
        }
    }
}

template <typename Cpu, typename Float, typename EdgeValue>
struct louvain_kernel {
    vertex_partitioning_result<task::vertex_partitioning> operator()(
        const detail::descriptor_base<task::vertex_partitioning>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const std::int32_t* init_partition,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        {
            using value_type = EdgeValue;
            using vertex_type = std::int32_t;
            using vertex_size_type = std::int64_t;
            using vertex_pointer_type = vertex_type*;

            using value_allocator_type = inner_alloc<value_type>;
            using vertex_allocator_type = inner_alloc<vertex_type>;
            using vertex_size_allocator_type = inner_alloc<vertex_size_type>;
            using vertex_pointer_allocator_type = inner_alloc<vertex_pointer_type>;

            using v1s_t = vector_container<vertex_size_type, vertex_size_allocator_type>;
            using v1p_t = vector_container<vertex_pointer_type, vertex_pointer_allocator_type>;

            vertex_allocator_type vertex_allocator(alloc_ptr);
            vertex_size_allocator_type vertex_size_allocator(alloc_ptr);
            value_allocator_type value_allocator(alloc_ptr);
            vertex_pointer_allocator_type vp_a(alloc_ptr);

            const Float resolution = static_cast<Float>(desc.get_resolution());
            const Float accuracy_threshold = static_cast<Float>(desc.get_accuracy_threshold());
            const std::int64_t max_iteration_count = desc.get_max_iteration_count();

            const std::int64_t vertex_count = t.get_vertex_count();
            const std::int64_t edge_count = t.get_edge_count();
            dal::preview::detail::topology<vertex_type> current_topology;

            vertex_size_type* current_topology_rows =
                allocate(vertex_size_allocator, vertex_count + 1);
            vertex_type* current_topology_cols = allocate(vertex_allocator, edge_count * 2);
            value_type* current_vals = allocate(value_allocator, edge_count * 2);
            value_type* current_self_loops = allocate(value_allocator, edge_count * 2);

            current_topology.set_topology(vertex_count,
                                          edge_count,
                                          current_topology_rows,
                                          current_topology_cols,
                                          edge_count * 2,
                                          nullptr);

            current_topology_rows[0] = t._rows_ptr[0];
            for (std::int64_t index = 0; index < vertex_count; index++) {
                current_topology_rows[index + 1] = t._rows_ptr[index + 1];
                current_self_loops[index] = 0;
            }
            for (std::int64_t index = 0; index < edge_count * 2; index++) {
                current_topology_cols[index] = t._cols_ptr[index];
                current_vals[index] = vals[index];
            }

            louvain_data<vertex_type, value_type> ld(vertex_count,
                                                     edge_count,
                                                     value_allocator,
                                                     vertex_allocator,
                                                     vertex_size_allocator);

            v1p_t communities(vp_a);
            v1s_t labels_size(vertex_size_allocator);
            v1s_t vertex_size(vertex_size_allocator);

            Float modularity = std::numeric_limits<Float>::min();
            vertex_type* labels = allocate(vertex_allocator, vertex_count);
            if (init_partition != nullptr) {
                for (std::int64_t v = 0; v < vertex_count; v++) {
                    labels[v] = init_partition[v];
                }
            }
            else {
                singleton_partition(labels, vertex_count);
            }

            bool allocate_labels = false;
            for (std::int64_t iteration = 0;
                 iteration < max_iteration_count || !max_iteration_count;
                 iteration++) {
                if (allocate_labels) {
                    labels = allocate(vertex_allocator, current_topology._vertex_count);
                    singleton_partition(labels, current_topology._vertex_count);
                }
                allocate_labels = true;
                bool changed = false;
                modularity = move_nodes<Cpu, Float>(current_topology,
                                                    current_vals,
                                                    current_self_loops,
                                                    labels,
                                                    changed,
                                                    resolution,
                                                    accuracy_threshold,
                                                    ld);

                if (!changed) {
                    deallocate(vertex_allocator, labels, current_topology._vertex_count);
                    break;
                }
                std::int64_t community_count = reindex_communities(labels,
                                                                   ld.community_size,
                                                                   current_topology._vertex_count,
                                                                   ld.index);

                compress_graph(current_topology,
                               current_vals,
                               current_self_loops,
                               community_count,
                               labels,
                               ld);
                labels_size.push_back(community_count);
                vertex_size.push_back(current_topology._vertex_count);
                communities.push_back(labels);
                current_topology._vertex_count = community_count;
            }

            auto labels_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* labels_ = labels_arr.get_mutable_data();
            set_result_labels(communities, vertex_size, init_partition, vertex_count, labels_);

            deallocate(vertex_size_allocator, current_topology_rows, vertex_count + 1);
            deallocate(vertex_allocator, current_topology_cols, edge_count * 2);
            deallocate(value_allocator, current_vals, edge_count * 2);
            deallocate(value_allocator, current_self_loops, edge_count * 2);

            if (communities.size() > 0) {
                deallocate(vertex_allocator, communities[0], vertex_count);
                for (std::int64_t iteration = 1; iteration < communities.size(); iteration++) {
                    deallocate(vertex_allocator,
                               communities[iteration],
                               labels_size[iteration - 1]);
                }
            }

            return vertex_partitioning_result<task::vertex_partitioning>()
                .set_labels(dal::detail::homogen_table_builder{}
                                .reset(labels_arr, t.get_vertex_count(), 1)
                                .build())
                .set_modularity(static_cast<double>(modularity))
                .set_community_count(current_topology._vertex_count);
        }
    }
};

} // namespace oneapi::dal::preview::louvain::backend
