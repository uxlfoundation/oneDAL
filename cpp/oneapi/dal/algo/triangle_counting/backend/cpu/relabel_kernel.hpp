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
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Index>
inline Index min(const Index& a, const Index& b) {
    return (a >= b) ? b : a;
}

template <typename Cpu>
void sort_ids_by_degree(const std::int32_t* degrees,
                        std::pair<std::int32_t, std::size_t>* degree_id_pairs, //Why std::size_t
                        std::int64_t vertex_count) {
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t n) {
        degree_id_pairs[n] = std::make_pair(degrees[n], (std::size_t)n);
    });
    dal::detail::parallel_sort(degree_id_pairs, degree_id_pairs + vertex_count);
    dal::detail::threader_for(vertex_count / 2, vertex_count / 2, [&](std::int64_t i) {
        std::swap(degree_id_pairs[i], degree_id_pairs[vertex_count - i - 1]);
    });
}

template <typename Cpu>
void fill_new_degrees_and_ids(const std::pair<std::int32_t, std::size_t>* degree_id_pairs,
                              std::int32_t* new_ids,
                              std::int32_t* degrees_relabel,
                              std::int64_t vertex_count) {
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t n) {
        const auto [degree, id] = degree_id_pairs[n];
        degrees_relabel[n] = degree;
        new_ids[id] = n;
    });
}

template <typename Cpu>
void parallel_prefix_sum(const std::int32_t* degrees_relabel,
                         std::int64_t* offsets,
                         std::int64_t* part_prefix,
                         std::int64_t* local_sums,
                         std::int64_t block_size,
                         std::int64_t num_blocks,
                         std::int64_t vertex_count) {
    dal::detail::threader_for(num_blocks, num_blocks, [&](std::int64_t block) {
        std::int64_t local_sum = 0;
        std::int64_t block_end = min((std::int64_t)((block + 1) * block_size), vertex_count);
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : local_sum))
        for (std::int64_t i = block * block_size; i < block_end; i++) {
            local_sum += degrees_relabel[i];
        }
        local_sums[block] = local_sum;
    });

    constexpr unsigned int simd_width = 4;
    if (num_blocks < simd_width + 1) {
        std::int64_t total = 0;
        for (std::int64_t block = 0; block < num_blocks; block++) {
            part_prefix[block] = total;
            total += local_sums[block];
        }
    }
    else {
        part_prefix[0] = 0;
        // TODO: vectorize with _mm_bslli_si128
        PRAGMA_OMP_SIMD
        for (std::int64_t block = 1; block < num_blocks - simd_width + 1; block += simd_width) {
            const std::int64_t shuffle1[simd_width] = { 0,
                                                        local_sums[block - 1],
                                                        local_sums[block],
                                                        local_sums[block + 1] };
            const std::int64_t shuffle2[simd_width] = { 0,
                                                        0,
                                                        local_sums[block - 1],
                                                        local_sums[block - 1] + local_sums[block] };
            part_prefix[block + 0] = part_prefix[block - 1] + local_sums[block - 1];
            part_prefix[block + 1] = part_prefix[block - 1] + local_sums[block + 0];
            part_prefix[block + 2] = part_prefix[block - 1] + local_sums[block + 1];
            part_prefix[block + 3] = part_prefix[block - 1] + local_sums[block + 2];
            part_prefix[block + 0] += shuffle1[0] + shuffle2[0];
            part_prefix[block + 1] += shuffle1[1] + shuffle2[1];
            part_prefix[block + 2] += shuffle1[2] + shuffle2[2];
            part_prefix[block + 3] += shuffle1[3] + shuffle2[3];
        }
        for (std::int64_t block = ((num_blocks + 2) / simd_width - 1) * simd_width;
             block < num_blocks;
             block++) {
            part_prefix[block] = part_prefix[block - 1] + local_sums[block - 1];
        }
    }
    part_prefix[num_blocks] = part_prefix[num_blocks - 1] + local_sums[num_blocks - 1];

    dal::detail::threader_for(num_blocks, num_blocks, [&](std::int64_t block) {
        std::int64_t local_total = part_prefix[block];
        std::int64_t block_end =
            min((std::int64_t)((block + 1) * block_size), (std::int64_t)vertex_count);
        for (std::int64_t i = block * block_size; i < block_end; i++) {
            offsets[i] = local_total;
            local_total += degrees_relabel[i];
        }
    });

    offsets[vertex_count] = part_prefix[num_blocks];
}

template <typename Cpu>
void fill_relabeled_topology(const dal::preview::detail::topology<std::int32_t>& t,
                             std::int32_t* vertex_neighbors_relabel,
                             std::int64_t* edge_offsets_relabel,
                             std::int64_t* offsets,
                             const std::int32_t* new_ids) {
    const auto vertex_count = t.get_vertex_count();
    dal::detail::threader_for(vertex_count + 1, vertex_count + 1, [&](std::int64_t n) {
        edge_offsets_relabel[n] = offsets[n];
    });

    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t u) {
        for (const std::int32_t* v = t.get_vertex_neighbors_begin(u);
             v != t.get_vertex_neighbors_end(u);
             ++v) {
            vertex_neighbors_relabel[offsets[new_ids[u]]++] = new_ids[*v];
        }

        dal::detail::parallel_sort(vertex_neighbors_relabel + edge_offsets_relabel[new_ids[u]],
                                   vertex_neighbors_relabel + edge_offsets_relabel[new_ids[u] + 1]);
    });
}

} // namespace oneapi::dal::preview::triangle_counting::backend
