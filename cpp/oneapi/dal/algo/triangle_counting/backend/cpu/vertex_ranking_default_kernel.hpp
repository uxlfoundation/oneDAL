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
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/algo/triangle_counting/backend/cpu/intersection_tc.hpp"
#include "oneapi/dal/backend/primitives/intersection/intersection.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Cpu>
array<std::int64_t> triangle_counting_local(const dal::preview::detail::topology<std::int32_t>& t,
                                            std::int64_t* triangles_local) {
    const auto vertex_count = t.get_vertex_count();
    //std::int32_t average_degree = t.get_edge_count() / vertex_count;
    int thread_cnt = dal::detail::threader_get_max_threads();

    dal::detail::threader_for(thread_cnt * vertex_count,
                              thread_cnt * vertex_count,
                              [&](std::int64_t u) {
                                  triangles_local[u] = 0;
                              });

    //const std::int32_t average_degree_sparsity_boundary = 4;
    if (true) {
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t u) {
            for (auto v_ = t.get_vertex_neighbors_begin(u); v_ != t.get_vertex_neighbors_end(u);
                 ++v_) {
                std::int32_t v = *v_;
                if (v > u) {
                    break;
                }
                auto u_neighbors_ptr = t.get_vertex_neighbors_begin(u);
                for (auto w_ = t.get_vertex_neighbors_begin(v); v_ != t.get_vertex_neighbors_end(v);
                     ++w_) {
                    std::int32_t w = *w_;
                    if (w > v) {
                        break;
                    }
                    while (*u_neighbors_ptr < w) {
                        u_neighbors_ptr++;
                    }
                    if (w == *u_neighbors_ptr) {
                        int thread_id = dal::detail::threader_get_current_thread_index();
                        std::int64_t indx = (std::int64_t)thread_id * (std::int64_t)vertex_count;
                        triangles_local[indx + u]++;
                        triangles_local[indx + v]++;
                        triangles_local[indx + w]++;
                    }
                }
            }
        });
    }
    else { //average_degree >= average_degree_sparsity_boundary
        // dal::detail::threader_for_simple(vertex_count, vertex_count, [&](std::int32_t u) {
        //     if (t.get_vertex_degree(u) >= 2)
        //         dal::detail::threader_for_int64ptr(
        //             static_cast<const std::int64_t*>(static_cast<const void*>(t.get_vertex_neighbors_begin(u))),
        //             static_cast<const std::int64_t*>(static_cast<const void*>(t.get_vertex_neighbors_end(u))),
        //             [&](const std::int32_t* v_) {
        //                 std::int32_t v = *v_;
        //                 if (v <= u) {
        //                     const std::int32_t u_degree = t.get_vertex_degree(u);
        //                     const std::int32_t* v_neighbors_begin = t.get_vertex_neighbors_begin(v);
        //                     const std::int32_t v_degree = t.get_vertex_degree(v);
        //                     std::int32_t new_v_degree;

        //                     for (new_v_degree = 0; (new_v_degree < v_degree) &&
        //                                            (v_neighbors_begin[new_v_degree] <= v);
        //                          new_v_degree++)
        //                         ;

        //                     int thread_id = dal::detail::threader_get_current_thread_index();
        //                     std::int64_t indx =
        //                         (std::int64_t)thread_id * (std::int64_t)vertex_count;

        //                     auto tc = intersection_local_tc<Cpu>{}(t.get_vertex_neighbors_begin(u),
        //                                                            t.get_vertex_neighbors_begin(v),
        //                                                            u_degree,
        //                                                            new_v_degree,
        //                                                            triangles_local + indx,
        //                                                            vertex_count);

        //                     triangles_local[indx + u] += tc;
        //                     triangles_local[indx + v] += tc;
        //                 }
        //             });
        // });
    }

    auto arr_triangles = array<std::int64_t>::empty(vertex_count);

    std::int64_t* triangles_ptr = arr_triangles.get_mutable_data();

    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t u) {
        triangles_ptr[u] = 0;
    });

    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t u) {
        for (int j = 0; j < thread_cnt; j++) {
            std::int64_t idx_glob = (std::int64_t)j * (std::int64_t)vertex_count;
            triangles_ptr[u] += triangles_local[idx_glob + u];
        }
    });
    return arr_triangles;
}

template <typename Cpu>
std::int64_t triangle_counting_global_scalar(
    const dal::preview::detail::topology<std::int32_t>& t) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        t.get_vertex_count(),
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                for (auto v_ = t.get_vertex_neighbors_begin(u); v_ != t.get_vertex_neighbors_end(u);
                     ++v_) {
                    std::int32_t v = *v_;
                    if (v > u) {
                        break;
                    }
                    auto u_neighbors_ptr = t.get_vertex_neighbors_begin(u);
                    for (auto w_ = t.get_vertex_neighbors_begin(v);
                         v_ != t.get_vertex_neighbors_end(u);
                         ++w_) {
                        std::int32_t w = *w_;
                        if (w > v) {
                            break;
                        }
                        while (*u_neighbors_ptr < w) {
                            u_neighbors_ptr++;
                        }
                        if (w == *u_neighbors_ptr) {
                            tc_u++;
                        }
                    }
                }
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

template <typename Cpu>
std::int64_t triangle_counting_global_vector(
    const dal::preview::detail::topology<std::int32_t>& t) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t_simple(
        t.get_vertex_count(),
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                if (t.get_vertex_degree(u) < 2) {
                    continue;
                }
                const auto u_neighbors_begin = t.get_vertex_neighbors_begin(u);
                const std::int32_t u_degree = t.get_vertex_degree(u);

                tc_u += oneapi::dal::detail::parallel_reduce_int32ptr_int64_t_simple(
                    t.get_vertex_neighbors_begin(u),
                    t.get_vertex_neighbors_end(u),
                    (std::int64_t)0,
                    [&](const std::int32_t* begin_v,
                        const std::int32_t* end_v,
                        std::int64_t total) -> std::int64_t {
                        for (auto v_ = begin_v; v_ != end_v; ++v_) {
                            std::int32_t v = *v_;

                            if (v > u) {
                                break;
                            }

                            const auto v_neighbors_begin = t.get_vertex_neighbors_begin(v);
                            const std::int32_t v_degree = t.get_vertex_degree(v);

                            std::int32_t new_v_degree = 0;
                            for (new_v_degree = 0; (new_v_degree < v_degree) &&
                                                   (v_neighbors_begin[new_v_degree] <= v);
                                 new_v_degree++)
                                ;

                            total += preview::backend::intersection<Cpu>(u_neighbors_begin,
                                                                         v_neighbors_begin,
                                                                         u_degree,
                                                                         new_v_degree);
                        }
                        return total;
                    },
                    [&](std::int64_t x, std::int64_t y) -> std::int64_t {
                        return x + y;
                    });
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

template <typename Cpu>
std::int64_t triangle_counting_global_vector_relabel(const std::int32_t* vertex_neighbors,
                                                     const std::int64_t* edge_offsets,
                                                     const std::int32_t* degrees,
                                                     std::int64_t vertex_count,
                                                     std::int64_t edge_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t_simple(
        vertex_count,
        (std::int64_t)0,
        [&](std::int64_t begin_u, std::int64_t end_u, std::int64_t tc_u) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                if (degrees[u] < 2) {
                    continue;
                }
                const std::int32_t* neigh_u = vertex_neighbors + edge_offsets[u];
                std::int32_t size_neigh_u = vertex_neighbors + edge_offsets[u + 1] - neigh_u;

                for (auto v_ = vertex_neighbors + edge_offsets[u];
                     v_ != vertex_neighbors + edge_offsets[u + 1];
                     ++v_) {
                    std::int32_t v = *v_;

                    if (v > u) {
                        break;
                    }

                    const std::int32_t* neigh_v = vertex_neighbors + edge_offsets[v];
                    std::int32_t size_neigh_v = vertex_neighbors + edge_offsets[v + 1] - neigh_v;

                    std::int32_t new_v_degree = 0;
                    for (new_v_degree = 0;
                         (new_v_degree < size_neigh_v) && (neigh_v[new_v_degree] <= v);
                         new_v_degree++)
                        ;

                    tc_u += preview::backend::intersection<Cpu>(neigh_u,
                                                                neigh_v,
                                                                size_neigh_u,
                                                                new_v_degree);
                }
            }
            return tc_u;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    return total_s;
}

template <typename Cpu>
std::int64_t compute_global_triangles(const array<std::int64_t>& local_triangles,
                                      std::int64_t vertex_count) {
    std::int64_t total_s = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        vertex_count,
        (std::int64_t)0,
        [&](std::int32_t begin_u, std::int32_t end_u, std::int64_t tc) -> std::int64_t {
            for (auto u = begin_u; u != end_u; ++u) {
                tc += local_triangles[u];
            }
            return tc;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return x + y;
        });
    total_s /= 3;
    return total_s;
}

} // namespace oneapi::dal::preview::triangle_counting::backend
