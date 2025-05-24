/* file: service_functions.hpp */
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

/// @file
/// Contains the service functionality for the graph objects

#pragma once

#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"

namespace oneapi::dal::preview {

/// Returns the number of vertices in the graph
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
/// @return The number of vertices in the graph
template <typename Graph>
constexpr auto get_vertex_count(const Graph &g) noexcept -> vertex_size_type<Graph>;

/// Returns the number of edges in the graph
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
///
/// @return The number of edges in the graph
template <typename Graph>
constexpr auto get_edge_count(const Graph &g) noexcept -> edge_size_type<Graph>;

/// Returns the degree for the specified vertex
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
/// @param [in]   u Vertex index
///
/// @return The degree of the vertex u
template <typename Graph>
constexpr auto get_vertex_degree(const Graph &g, vertex_type<Graph> u)
    -> vertex_edge_size_type<Graph>;

/// Returns the outward degree for the specified vertex
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
/// @param [in]   u Vertex index
///
/// @return The outward degree of the vertex
template <typename Graph>
constexpr auto get_vertex_outward_degree(const Graph &g, vertex_type<Graph> u)
    -> vertex_outward_edge_size_type<Graph>;

/// Returns the range of the vertex neighbors for the specified vertex
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
/// @param [in]   u Vertex index
///
/// @return The range of the vertex u neighbors
template <typename Graph>
constexpr auto get_vertex_neighbors(const Graph &g, vertex_type<Graph> u)
    -> const_vertex_edge_range_type<Graph>;

/// Returns the range of the vertex outward neighbors for the specified vertex
///
/// @tparam Graph  Type of the graph
/// @param [in]   g  Input graph object
/// @param [in]   u Vertex index
///
/// @return The range of the vertex out neighbors
template <typename Graph>
constexpr auto get_vertex_outward_neighbors(const Graph &g, vertex_type<Graph> u)
    -> const_vertex_outward_edge_range_type<Graph>;

/// Returns the value of an edge (u, v)
///
/// @tparam Graph  Type of the graph
/// @param [in]   g Input graph object
/// @param [in]   u Source vertex index
/// @param [in]   v Destination vertex index
///
/// @return Edge value
template <typename Graph>
constexpr auto get_edge_value(const Graph &g, vertex_type<Graph> u, vertex_type<Graph> v)
    -> const edge_user_value_type<Graph> &;

//Functions implementation
template <typename Graph>
constexpr auto get_vertex_count(const Graph &g) noexcept -> vertex_size_type<Graph> {
    return detail::get_vertex_count_impl(g);
}

template <typename Graph>
constexpr auto get_edge_count(const Graph &g) noexcept -> edge_size_type<Graph> {
    return detail::get_edge_count_impl(g);
}

template <typename Graph>
constexpr auto get_vertex_degree(const Graph &g, vertex_type<Graph> u)
    -> vertex_edge_size_type<Graph> {
    static_assert(!is_directed<Graph>, "get_vertex_degree requires graph undirectness");
    if (u < 0 || (vertex_size_type<Graph>)u >= detail::get_vertex_count_impl(g)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_degree_impl(g, u);
}

template <typename Graph>
constexpr auto get_vertex_outward_degree(const Graph &g, vertex_type<Graph> u)
    -> vertex_outward_edge_size_type<Graph> {
    static_assert(is_directed<Graph>, "get_vertex_out_degree requires graph directness");
    if (u < 0 || (vertex_size_type<Graph>)u >= detail::get_vertex_count_impl(g)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_outward_degree_impl(g, u);
}

template <typename Graph>
constexpr auto get_vertex_neighbors(const Graph &g, vertex_type<Graph> u)
    -> const_vertex_edge_range_type<Graph> {
    static_assert(!is_directed<Graph>, "get_vertex_neighbors requires graph undirectness");
    if (u < 0 || (vertex_size_type<Graph>)u >= detail::get_vertex_count_impl(g)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_neighbors_impl(g, u);
}

template <typename Graph>
constexpr auto get_vertex_outward_neighbors(const Graph &g, vertex_type<Graph> u)
    -> const_vertex_outward_edge_range_type<Graph> {
    static_assert(is_directed<Graph>, "get_vertex_out_neighbors requires graph directness");
    if (u < 0 || (vertex_size_type<Graph>)u >= detail::get_vertex_count_impl(g)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_outward_neighbors_impl(g, u);
}

template <typename Graph>
constexpr auto get_edge_value(const Graph &g, vertex_type<Graph> u, vertex_type<Graph> v)
    -> const edge_user_value_type<Graph> & {
    static_assert(is_directed<Graph>, "get_edge_value supported only for directed graph");
    if (u < 0 || (vertex_size_type<Graph>)u >= detail::get_vertex_count_impl(g)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_edge_value_impl(g, u, v);
}

} // namespace oneapi::dal::preview
