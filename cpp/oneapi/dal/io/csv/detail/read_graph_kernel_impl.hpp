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

#include <algorithm>
#include <atomic>
#include <fstream>

#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv/detail/read_graph_service.hpp"
#include "oneapi/dal/io/csv/detail/common.hpp"

namespace oneapi::dal::preview::csv::detail {

template <typename EdgeList>
inline void read_edge_list(const std::string &name, EdgeList &elist);

template <typename Vertex>
inline void read_edge_list(const std::string &name, edge_list<Vertex> &elist) {
    std::ifstream file(name);
    if (!file.is_open()) {
        throw invalid_argument(dal::detail::error_messages::file_not_found());
    }
    elist.reserve(1024);

    std::string line;
    char *source_endptr;
    char *dest_endptr;
    const char *edgeline;
    Vertex source_vertex;
    Vertex destination_vertex;

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        edgeline = line.c_str();
        source_vertex = daal_string_to_int(edgeline, &source_endptr);
        if (std::isspace(*source_endptr)) {
            source_endptr++;
        }
        destination_vertex = daal_string_to_int(source_endptr, &dest_endptr);

        if (source_endptr == dest_endptr) {
            throw invalid_argument("Invalid line content: " + line);
        }

        if (source_vertex < 0 || destination_vertex < 0) {
            throw invalid_argument("Negative vertex ids: " + line);
        }

        if (*dest_endptr != '\0') {
            while (*dest_endptr != '\0') {
                if (!std::isspace(*dest_endptr)) {
                    throw invalid_argument("Invalid line content: " + line);
                }
                dest_endptr++;
            }
        }

        auto edge = std::make_pair(source_vertex, destination_vertex);
        elist.push_back(edge);
    }
    file.close();
}

template <typename Vertex, typename Weight>
inline void read_edge_list(const std::string &name, weighted_edge_list<Vertex, Weight> &elist) {
    std::ifstream file(name);
    if (!file.is_open()) {
        throw invalid_argument(dal::detail::error_messages::file_not_found());
    }
    elist.reserve(1024);

    std::string line;
    char *source_endptr;
    char *dest_endptr;
    char *value_endptr;
    const char *edgeline;
    Vertex source_vertex;
    Vertex destination_vertex;
    Weight edge_value;

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        edgeline = line.c_str();
        source_vertex = daal_string_to<Vertex>(edgeline, &source_endptr);
        if (std::isspace(*source_endptr)) {
            source_endptr++;
        }
        destination_vertex = daal_string_to<Vertex>(source_endptr, &dest_endptr);
        if (std::isspace(*dest_endptr)) {
            dest_endptr++;
        }
        edge_value = daal_string_to<Weight>(dest_endptr, &value_endptr);

        if (dest_endptr == value_endptr) {
            throw invalid_argument("Invalid line content: " + line);
        }

        if (source_vertex < 0 || destination_vertex < 0) {
            throw invalid_argument("Negative vertex ids: " + line);
        }

        if (*value_endptr != '\0') {
            while (*value_endptr != '\0') {
                if (!std::isspace(*value_endptr)) {
                    throw invalid_argument("Invalid line content: " + line);
                }
                value_endptr++;
            }
        }

        auto edge =
            std::tuple<Vertex, Vertex, Weight>(source_vertex, destination_vertex, edge_value);
        elist.push_back(edge);
    }
    file.close();
}

template <typename EdgeList>
std::int64_t get_vertex_count_from_edge_list(const EdgeList &edges) {
    auto max_id = std::get<0>(edges[0]);
    for (std::int64_t i = 0; i < edges.size(); i++) {
        auto edge_max = std::max(std::get<0>(edges[i]), std::get<1>(edges[i]));
        max_id = std::max(max_id, edge_max);
    }

    const std::int64_t vertex_count = max_id + 1;
    return vertex_count;
}

template <typename Graph, bool IsDirected = is_directed<Graph>>
struct collect_degrees_from_edge_list;

template <typename Graph>
struct collect_degrees_from_edge_list<Graph, /* IsDirected = */ false> {
    template <typename EdgeList, typename AtomicType>
    auto operator()(const EdgeList &edges, AtomicType *degrees_cv) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            ++degrees_cv[std::get<0>(edges[u])];
            ++degrees_cv[std::get<1>(edges[u])];
        });
    }
};

template <typename Graph>
struct collect_degrees_from_edge_list<Graph, /* IsDirected = */ true> {
    template <typename EdgeList, typename AtomicType>
    auto operator()(const EdgeList &edges, AtomicType *degrees_cv) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            ++degrees_cv[std::get<0>(edges[u])];
        });
    }
};

template <typename Graph, bool IsDirected = is_directed<Graph>>
struct get_edges_count;

template <typename Graph>
struct get_edges_count<Graph, /* IsDirected = */ false> {
    template <typename Index>
    Index operator()(Index degrees_sum) {
        return degrees_sum / 2;
    }
};

template <typename Graph>
struct get_edges_count<Graph, /* IsDirected = */ true> {
    template <typename Index>
    Index operator()(Index degrees_sum) {
        return degrees_sum;
    }
};

template <typename EdgeIndex, typename AtomicVertex, typename AtomicEdge>
EdgeIndex compute_prefix_sum_atomic(const AtomicVertex *degrees,
                                    std::int64_t degrees_count,
                                    AtomicEdge *edge_offsets_atomic) {
    EdgeIndex total_sum_degrees = 0;
    edge_offsets_atomic[0] = total_sum_degrees;
    for (std::int64_t i = 0; i < degrees_count; ++i) {
        total_sum_degrees += degrees[i].load();
        edge_offsets_atomic[i + 1] = total_sum_degrees;
    }
    return total_sum_degrees;
}

template <typename EdgeIndex, typename VertexIndex>
EdgeIndex compute_prefix_sum(const VertexIndex *degrees,
                             std::int64_t degrees_count,
                             EdgeIndex *edge_offsets) {
    EdgeIndex total_sum_degrees = 0;
    edge_offsets[0] = total_sum_degrees;
    for (std::int64_t i = 0; i < degrees_count; ++i) {
        total_sum_degrees += degrees[i];
        edge_offsets[i + 1] = total_sum_degrees;
    }
    return total_sum_degrees;
}

template <>
ONEDAL_EXPORT std::int64_t compute_prefix_sum<std::int64_t, std::int32_t>(
    const std::int32_t *degrees,
    std::int64_t degrees_count,
    std::int64_t *edge_offsets);

template <typename Index, typename AtomicIndex>
void fill_from_atomics(Index *arr, AtomicIndex *atomic_arr, std::int64_t elements_count) {
    dal::detail::threader_for_int64(elements_count, [&](std::int64_t n) {
        arr[n] = atomic_arr[n].load();
    });
}

template <typename Graph, bool IsDirected = is_directed<Graph>>
struct fill_unfiltered_neighs;

template <typename Graph>
struct fill_unfiltered_neighs<Graph, /* IsDirected = */ false> {
    template <typename Vertex, typename AtomicEdge>
    auto operator()(const edge_list<Vertex> &edges,
                    AtomicEdge *rows_vec_atomic,
                    Vertex *unfiltered_neighs) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            const auto [src, dst] = edges[u];
            unfiltered_neighs[++rows_vec_atomic[src] - 1] = dst;
            unfiltered_neighs[++rows_vec_atomic[dst] - 1] = src;
        });
    }

    template <typename Vertex, typename Weight, typename AtomicEdge>
    auto operator()(const weighted_edge_list<Vertex, Weight> &edges,
                    AtomicEdge *rows_vec_atomic,
                    std::pair<Vertex, Weight> *unfiltered_neighs_vals) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            unfiltered_neighs_vals[++rows_vec_atomic[std::get<0>(edges[u])] - 1] =
                std::make_pair(std::get<1>(edges[u]), std::get<2>(edges[u]));
            unfiltered_neighs_vals[++rows_vec_atomic[std::get<1>(edges[u])] - 1] =
                std::make_pair(std::get<0>(edges[u]), std::get<2>(edges[u]));
        });
    }
};

template <typename Graph>
struct fill_unfiltered_neighs<Graph, /* IsDirected = */ true> {
    template <typename Vertex, typename AtomicEdge>
    auto operator()(const edge_list<Vertex> &edges,
                    AtomicEdge *rows_vec_atomic,
                    Vertex *unfiltered_neighs) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            const auto [src, dst] = edges[u];
            unfiltered_neighs[++rows_vec_atomic[src] - 1] = dst;
        });
    }

    template <typename Vertex, typename Weight, typename AtomicEdge>
    auto operator()(const weighted_edge_list<Vertex, Weight> &edges,
                    AtomicEdge *rows_vec_atomic,
                    std::pair<Vertex, Weight> *unfiltered_neighs_vals) {
        dal::detail::threader_for_int64(edges.size(), [&](std::int64_t u) {
            unfiltered_neighs_vals[++rows_vec_atomic[std::get<0>(edges[u])] - 1] =
                std::make_pair(std::get<1>(edges[u]), std::get<2>(edges[u]));
        });
    }
};

template <typename VertexIndex, typename EdgeIndex>
void fill_filtered_neighs(const EdgeIndex *unfiltered_offsets,
                          const VertexIndex *unfiltered_neighs,
                          const VertexIndex *filtered_degrees,
                          const EdgeIndex *filtered_offsets,
                          VertexIndex *filtered_neighs,
                          std::int64_t vertex_count) {
    dal::detail::threader_for_int64(vertex_count, [&](std::int64_t u) {
        auto u_neighs = filtered_neighs + filtered_offsets[u];
        auto u_neighs_unf = unfiltered_neighs + unfiltered_offsets[u];
        for (VertexIndex i = 0; i < filtered_degrees[u]; i++) {
            u_neighs[i] = u_neighs_unf[i];
        }
    });
}

template <typename Vertex, typename Weight, typename Edge>
void fill_filtered_neighs(const Edge *unfiltered_offsets,
                          const std::pair<Vertex, Weight> *unfiltered_neighs_vals,
                          const Vertex *filtered_degrees,
                          const Edge *filtered_offsets,
                          Vertex *filtered_neighs,
                          Weight *filtered_vals,
                          std::int64_t vertex_count) {
    dal::detail::threader_for_int64(vertex_count, [&](std::int64_t u) {
        auto u_neighs = filtered_neighs + filtered_offsets[u];
        auto u_neighs_vals = filtered_vals + filtered_offsets[u];
        auto u_neighs_unf = unfiltered_neighs_vals + unfiltered_offsets[u];
        for (Vertex i = 0; i < filtered_degrees[u]; i++) {
            u_neighs[i] = std::get<0>(u_neighs_unf[i]);
            u_neighs_vals[i] = std::get<1>(u_neighs_unf[i]);
        }
    });
}

template <>
ONEDAL_EXPORT void fill_filtered_neighs<std::int32_t, std::int64_t>(
    const std::int64_t *unfiltered_offsets,
    const std::int32_t *unfiltered_neighs,
    const std::int32_t *filtered_degrees,
    const std::int64_t *filtered_offsets,
    std::int32_t *filtered_neighs,
    std::int64_t vertex_count);

template <typename VertexIndex, typename EdgeIndex>
void filter_neighbors_and_fill_new_degrees(VertexIndex *unfiltered_neighs,
                                           EdgeIndex *unfiltered_offsets,
                                           VertexIndex *new_degrees,
                                           std::int64_t vertex_count) {
    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted
    dal::detail::threader_for_int64(vertex_count, [&](std::int64_t u) {
        auto start_p = unfiltered_neighs + unfiltered_offsets[u];
        auto end_p = unfiltered_neighs + unfiltered_offsets[u + 1];

        //dal::detail::parallel_sort(start_p, end_p);
        std::sort(start_p, end_p);
        auto neighs_u_new_end = std::unique(start_p, end_p);
        neighs_u_new_end = std::remove(start_p, neighs_u_new_end, u);
        new_degrees[u] = (VertexIndex)std::distance(start_p, neighs_u_new_end);
    });
}

template <typename Vertex, typename Weight, typename EdgeIndex>
void filter_neighbors_and_fill_new_degrees(std::pair<Vertex, Weight> *unfiltered_neighs_vals,
                                           EdgeIndex *unfiltered_offsets,
                                           Vertex *new_degrees,
                                           std::int64_t vertex_count) {
    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted
    dal::detail::threader_for_int64(vertex_count, [&](std::int64_t u) {
        auto start_p = unfiltered_neighs_vals + unfiltered_offsets[u];
        auto end_p = unfiltered_neighs_vals + unfiltered_offsets[u + 1];

        std::sort(start_p, end_p);
        auto neighs_u_new_end =
            std::unique(start_p,
                        end_p,
                        [](std::pair<Vertex, Weight> &p1, std::pair<Vertex, Weight> &p2) {
                            return (std::get<0>(p1) == std::get<0>(p2) ? true : false);
                        });
        neighs_u_new_end =
            std::remove_if(start_p, neighs_u_new_end, [&u](std::pair<Vertex, Weight> &p) {
                return (std::get<0>(p) == u ? true : false);
            });
        new_degrees[u] = (Vertex)std::distance(start_p, neighs_u_new_end);
    });
}

template <>
ONEDAL_EXPORT void filter_neighbors_and_fill_new_degrees<std::int32_t, std::int64_t>(
    std::int32_t *unfiltered_neighs,
    std::int64_t *unfiltered_offsets,
    std::int32_t *new_degrees,
    std::int64_t vertex_count);

template <typename Graph>
void convert_to_csr_impl(const edge_list<typename graph_traits<Graph>::vertex_type> &edges,
                         Graph &g) {
    if (edges.size() == 0) {
        throw invalid_argument(dal::detail::error_messages::empty_edge_list());
    }

    using vertex_t = typename graph_traits<Graph>::vertex_type;
    using vertex_size_type = typename graph_traits<Graph>::vertex_size_type;
    using edge_t = typename graph_traits<Graph>::edge_type;

    using atomic_vertex_t = typename std::atomic<vertex_t>;
    using atomic_edge_t = typename std::atomic<edge_t>;

    const vertex_size_type vertex_count = get_vertex_count_from_edge_list(edges);
    if (vertex_count < 0) {
        throw range_error(dal::detail::error_messages::overflow_found_in_sum_of_two_values());
    }

    auto &graph_impl = oneapi::dal::detail::get_impl(g);
    auto &allocator = graph_impl._allocator;

    using namespace oneapi::dal::preview::detail;

    rebinded_allocator ra(allocator);

    auto [degrees_cv_array, degrees_cv] =
        ra.template allocate_array<dal::array<atomic_vertex_t>>(vertex_count);
    degrees_cv = new (degrees_cv) atomic_vertex_t[vertex_count]();

    collect_degrees_from_edge_list<Graph>{}(edges, degrees_cv);

    const vertex_size_type rows_vec_count = vertex_count + 1;
    if ((rows_vec_count - vertex_count) != static_cast<vertex_size_type>(1)) {
        throw range_error(dal::detail::error_messages::overflow_found_in_sum_of_two_values());
    }

    auto [rows_vec_atomic_array, rows_vec_atomic] =
        ra.template allocate_array<dal::array<atomic_edge_t>>(rows_vec_count);
    rows_vec_atomic = new (rows_vec_atomic) atomic_edge_t[rows_vec_count]();

    edge_t total_sum_degrees =
        compute_prefix_sum_atomic<edge_t>(degrees_cv, vertex_count, rows_vec_atomic);

    degrees_cv_array.reset();

    auto [unfiltered_offsets_array, unfiltered_offsets] =
        ra.template allocate_array<dal::array<edge_t>>(rows_vec_count);

    fill_from_atomics(unfiltered_offsets, rows_vec_atomic, rows_vec_count);

    auto [unfiltered_neighs_array, unfiltered_neighs] =
        ra.template allocate_array<dal::array<vertex_t>>(total_sum_degrees);

    fill_unfiltered_neighs<Graph>{}(edges, rows_vec_atomic, unfiltered_neighs);

    rows_vec_atomic_array.reset();

    using edge_set_t = typename graph_traits<Graph>::edge_set;
    using vertex_set_t = typename graph_traits<Graph>::vertex_set;

    auto [degrees_array, degrees_data] = ra.template allocate_array<vertex_set_t>(vertex_count);

    filter_neighbors_and_fill_new_degrees(unfiltered_neighs,
                                          unfiltered_offsets,
                                          degrees_data,
                                          vertex_count);

    auto edge_offsets_tup = ra.template allocate_array<edge_set_t>(vertex_count + 1);
    auto edge_offsets_data = std::get<1>(edge_offsets_tup);
    auto rows_array = std::get<0>(edge_offsets_tup);

    edge_t filtered_total_sum_degrees =
        compute_prefix_sum(degrees_data, vertex_count, edge_offsets_data);

    auto [cols_array, vertex_neighbors] =
        ra.template allocate_array<vertex_set_t>(filtered_total_sum_degrees);

    fill_filtered_neighs(unfiltered_offsets,
                         unfiltered_neighs,
                         degrees_data,
                         edge_offsets_data,
                         vertex_neighbors,
                         vertex_count);

    unfiltered_neighs_array.reset();
    unfiltered_offsets_array.reset();
    graph_impl.set_topology(cols_array,
                            rows_array,
                            degrees_array,
                            get_edges_count<Graph>{}(filtered_total_sum_degrees));

    if (filtered_total_sum_degrees < oneapi::dal::detail::limits<std::int32_t>::max()) {
        using vertex_edge_t = typename graph_traits<Graph>::impl_type::vertex_edge_type;
        using vertex_edge_set_t = typename graph_traits<Graph>::impl_type::vertex_edge_set;

        oneapi::dal::preview::detail::rebinded_allocator ra_vertex_edge(
            graph_impl._vertex_edge_allocator);

        auto edge_offsets_tup =
            ra_vertex_edge.template allocate_array<vertex_edge_set_t>(vertex_count + 1);
        auto rows_vertex = std::get<1>(edge_offsets_tup);
        dal::detail::threader_for_int64(vertex_count + 1, [&](std::int64_t u) {
            rows_vertex[u] = static_cast<vertex_edge_t>(edge_offsets_data[u]);
        });

        graph_impl.get_topology()._rows_vertex = std::get<0>(edge_offsets_tup);
    }

    return;
}

template <typename Graph>
void convert_to_csr_impl(
    const weighted_edge_list<typename graph_traits<Graph>::vertex_type,
                             typename graph_traits<Graph>::edge_user_value_type> &edges,
    Graph &g) {
    if (edges.size() == 0) {
        throw invalid_argument(dal::detail::error_messages::empty_edge_list());
    }

    using vertex_t = typename graph_traits<Graph>::vertex_type;
    using vertex_size_type = typename graph_traits<Graph>::vertex_size_type;
    using edge_t = typename graph_traits<Graph>::edge_type;
    using edge_value_type = typename graph_traits<Graph>::edge_user_value_type;
    using vertex_weight_pair = std::pair<vertex_t, edge_value_type>;

    using atomic_vertex_t = typename std::atomic<vertex_t>;
    using atomic_edge_t = typename std::atomic<edge_t>;

    const vertex_size_type vertex_count = get_vertex_count_from_edge_list(edges);
    if (vertex_count < 0) {
        throw range_error(dal::detail::error_messages::overflow_found_in_sum_of_two_values());
    }

    auto &graph_impl = oneapi::dal::detail::get_impl(g);
    auto &allocator = graph_impl._allocator;

    using namespace oneapi::dal::preview::detail;

    rebinded_allocator ra(allocator);

    auto [degrees_cv_array, degrees_cv] =
        ra.template allocate_array<dal::array<atomic_vertex_t>>(vertex_count);
    degrees_cv = new (degrees_cv) atomic_vertex_t[vertex_count]();

    collect_degrees_from_edge_list<Graph>{}(edges, degrees_cv);

    const vertex_size_type rows_vec_count = vertex_count + 1;
    if ((rows_vec_count - vertex_count) != static_cast<vertex_size_type>(1)) {
        throw range_error(dal::detail::error_messages::overflow_found_in_sum_of_two_values());
    }

    auto [rows_vec_atomic_array, rows_vec_atomic] =
        ra.template allocate_array<dal::array<atomic_edge_t>>(rows_vec_count);
    rows_vec_atomic = new (rows_vec_atomic) atomic_edge_t[rows_vec_count]();

    edge_t total_sum_degrees =
        compute_prefix_sum_atomic<edge_t>(degrees_cv, vertex_count, rows_vec_atomic);

    degrees_cv_array.reset();

    auto [unfiltered_offsets_array, unfiltered_offsets] =
        ra.template allocate_array<dal::array<edge_t>>(rows_vec_count);

    fill_from_atomics(unfiltered_offsets, rows_vec_atomic, rows_vec_count);

    auto [unfiltered_neighs_and_vals_array, unfiltered_neighs_and_vals] =
        ra.template allocate_array<dal::array<vertex_weight_pair>>(total_sum_degrees);
    unfiltered_neighs_and_vals =
        new (unfiltered_neighs_and_vals) vertex_weight_pair[total_sum_degrees]();

    fill_unfiltered_neighs<Graph>{}(edges, rows_vec_atomic, unfiltered_neighs_and_vals);

    rows_vec_atomic_array.reset();

    using edge_set_t = typename graph_traits<Graph>::edge_set;
    using vertex_set_t = typename graph_traits<Graph>::vertex_set;

    auto [degrees_array, degrees_data] = ra.template allocate_array<vertex_set_t>(vertex_count);

    filter_neighbors_and_fill_new_degrees(unfiltered_neighs_and_vals,
                                          unfiltered_offsets,
                                          degrees_data,
                                          vertex_count);

    auto edge_offsets_tup = ra.template allocate_array<edge_set_t>(vertex_count + 1);
    auto edge_offsets_data = std::get<1>(edge_offsets_tup);
    auto rows_array = std::get<0>(edge_offsets_tup);

    edge_t filtered_total_sum_degrees =
        compute_prefix_sum(degrees_data, vertex_count, edge_offsets_data);

    auto [cols_array, vertex_neighbors] =
        ra.template allocate_array<vertex_set_t>(filtered_total_sum_degrees);

    auto [edge_values_array, edge_values_data] =
        ra.template allocate_array<edge_values<edge_value_type>>(filtered_total_sum_degrees);

    fill_filtered_neighs(unfiltered_offsets,
                         unfiltered_neighs_and_vals,
                         degrees_data,
                         edge_offsets_data,
                         vertex_neighbors,
                         edge_values_data,
                         vertex_count);

    unfiltered_neighs_and_vals_array.reset();
    unfiltered_offsets_array.reset();
    graph_impl.set_topology(cols_array,
                            rows_array,
                            degrees_array,
                            get_edges_count<Graph>{}(filtered_total_sum_degrees));
    graph_impl.set_edge_values(edge_values_array);

    if (filtered_total_sum_degrees < oneapi::dal::detail::limits<std::int32_t>::max()) {
        using vertex_edge_t = typename graph_traits<Graph>::impl_type::vertex_edge_type;
        using vertex_edge_set_t = typename graph_traits<Graph>::impl_type::vertex_edge_set;

        oneapi::dal::preview::detail::rebinded_allocator ra_vertex_edge(
            graph_impl._vertex_edge_allocator);

        auto edge_offsets_tup =
            ra_vertex_edge.template allocate_array<vertex_edge_set_t>(vertex_count + 1);
        auto rows_vertex = std::get<1>(edge_offsets_tup);
        dal::detail::threader_for_int64(vertex_count + 1, [&](std::int64_t u) {
            rows_vertex[u] = static_cast<vertex_edge_t>(edge_offsets_data[u]);
        });

        graph_impl.get_topology()._rows_vertex = std::get<0>(edge_offsets_tup);
    }

    return;
}

template <typename EdgeListType, typename Descriptor, typename DataSource>
void read_from_edgelist(EdgeListType &elist,
                        const DataSource &ds,
                        const Descriptor &desc,
                        typename Descriptor::object_t &graph) {
    auto alloc = desc.get_allocator();
    read_edge_list(ds.get_file_name(), elist);
    convert_to_csr_impl(elist, graph);
    return;
}

template <bool IsEdgeWeighted, typename Descriptor, typename DataSource>
class read_impl {};

template <typename Descriptor, typename DataSource>
class read_impl<false, Descriptor, DataSource> {
    using graph_t = typename Descriptor::object_t;
    using vertex_t = typename dal::preview::vertex_type<graph_t>;
    using edge_list_t = typename preview::edge_list<vertex_t>;
    edge_list_t elist;

public:
    void operator()(const DataSource &ds,
                    const Descriptor &desc,
                    typename Descriptor::object_t &graph) {
        read_from_edgelist(elist, ds, desc, graph);
    }
};

template <typename Descriptor, typename DataSource>
class read_impl<true, Descriptor, DataSource> {
    using graph_t = typename Descriptor::object_t;
    using vertex_t = typename dal::preview::vertex_type<graph_t>;
    using weight_t = typename dal::preview::edge_user_value_type<graph_t>;
    using edge_list_t = typename preview::weighted_edge_list<vertex_t, weight_t>;
    edge_list_t elist;

public:
    void operator()(const DataSource &ds,
                    const Descriptor &desc,
                    typename Descriptor::object_t &graph) {
        read_from_edgelist(elist, ds, desc, graph);
        return;
    }
};

} // namespace oneapi::dal::preview::csv::detail
