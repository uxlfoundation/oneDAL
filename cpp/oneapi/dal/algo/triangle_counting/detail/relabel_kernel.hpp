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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace oneapi::dal::preview::triangle_counting::detail {

ONEDAL_EXPORT void sort_ids_by_degree(const dal::detail::host_policy& policy,
                                      const std::int32_t* degrees,
                                      std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                                      std::int64_t vertex_count);

ONEDAL_EXPORT void fill_new_degrees_and_ids(
    const dal::detail::host_policy& policy,
    const std::pair<std::int32_t, std::size_t>* degree_id_pairs,
    std::int32_t* new_ids,
    std::int32_t* degrees_relabel,
    std::int64_t vertex_count);

ONEDAL_EXPORT void parallel_prefix_sum(const dal::detail::host_policy& policy,
                                       const std::int32_t* degrees_relabel,
                                       std::int64_t* offsets,
                                       std::int64_t* part_prefix,
                                       std::int64_t* local_sums,
                                       std::int64_t block_size,
                                       std::int64_t num_blocks,
                                       std::int64_t vertex_count);

ONEDAL_EXPORT void fill_relabeled_topology(const dal::detail::host_policy& policy,
                                           const dal::preview::detail::topology<std::int32_t>& t,
                                           std::int32_t* vertex_neighbors_relabel,
                                           std::int64_t* edge_offsets_relabel,
                                           std::int64_t* offsets,
                                           const std::int32_t* new_ids);

template <typename Allocator>
struct relabel_by_greater_degree {
    inline void operator()(const dal::detail::host_policy& ctx,
                           const dal::preview::detail::topology<std::int32_t>& t,
                           std::int32_t* vertex_neighbors_relabel,
                           std::int64_t* edge_offsets_relabel,
                           std::int32_t* degrees_relabel,
                           const Allocator& alloc) {
        const auto vertex_count = t.get_vertex_count();
        using int32_allocator_type =
            typename std::allocator_traits<Allocator>::template rebind_alloc<std::int32_t>;

        using pair_allocator_type = typename std::allocator_traits<
            Allocator>::template rebind_alloc<std::pair<std::int32_t, std::size_t>>;

        using int64_allocator_type =
            typename std::allocator_traits<Allocator>::template rebind_alloc<std::int64_t>;

        int64_allocator_type int64_allocator(alloc);
        pair_allocator_type pair_allocator(alloc);
        int32_allocator_type int32_allocator(alloc);

        std::pair<std::int32_t, std::size_t>* degree_id_pairs =

            oneapi::dal::preview::detail::allocate(pair_allocator, vertex_count);

        sort_ids_by_degree(ctx, t._degrees.get_data(), degree_id_pairs, vertex_count);

        std::int32_t* new_ids =
            oneapi::dal::preview::detail::allocate(int32_allocator, vertex_count);

        fill_new_degrees_and_ids(ctx, degree_id_pairs, new_ids, degrees_relabel, vertex_count);

        oneapi::dal::preview::detail::deallocate(pair_allocator, degree_id_pairs, vertex_count);

        std::int64_t* offsets =
            oneapi::dal::preview::detail::allocate(int64_allocator, vertex_count + 1);

        constexpr std::size_t block_size = 1 << 20;
        constexpr std::int64_t block_size_int64 = std::int64_t(block_size);
        dal::detail::check_sum_overflow(vertex_count, block_size_int64 - 1);
        const std::int64_t num_blocks = (vertex_count + block_size_int64 - 1) / block_size_int64;
        std::int64_t* local_sums =
            oneapi::dal::preview::detail::allocate(int64_allocator, num_blocks);
        std::int64_t* part_prefix =
            oneapi::dal::preview::detail::allocate(int64_allocator, num_blocks + 1);

        parallel_prefix_sum(ctx,
                            degrees_relabel,
                            offsets,
                            part_prefix,
                            local_sums,
                            block_size,
                            num_blocks,
                            vertex_count);

        oneapi::dal::preview::detail::deallocate(int64_allocator, local_sums, num_blocks);
        oneapi::dal::preview::detail::deallocate(int64_allocator, part_prefix, num_blocks + 1);

        fill_relabeled_topology(ctx,
                                t,
                                vertex_neighbors_relabel,
                                edge_offsets_relabel,
                                offsets,
                                new_ids);

        oneapi::dal::preview::detail::deallocate(int64_allocator, offsets, vertex_count + 1);
        oneapi::dal::preview::detail::deallocate(int32_allocator, new_ids, vertex_count);
    }
};

} // namespace oneapi::dal::preview::triangle_counting::detail
