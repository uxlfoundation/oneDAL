/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <array>
#include <set>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/connected_components/vertex_partitioning.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::connected_components::gpu::test {

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_component_count() const {
        return component_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
    std::int64_t component_count;
};

class triangle_graph_type : public graph_base_data {
public:
    triangle_graph_type() {
        vertex_count = 3;
        edge_count = 3;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        component_count = 1;
    }

    std::array<std::int32_t, 3> degrees = { 2, 2, 2 };
    std::array<std::int32_t, 6> cols = { 1, 2, 0, 2, 0, 1 };
    std::array<std::int64_t, 4> rows = { 0, 2, 4, 6 };
};

class two_components_graph_type : public graph_base_data {
public:
    two_components_graph_type() {
        vertex_count = 4;
        edge_count = 2;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        component_count = 2;
    }

    std::array<std::int32_t, 4> degrees = { 1, 1, 1, 1 };
    std::array<std::int32_t, 4> cols = { 1, 0, 3, 2 };
    std::array<std::int64_t, 5> rows = { 0, 1, 2, 3, 4 };
};

class complete_graph_5_type : public graph_base_data {
public:
    complete_graph_5_type() {
        vertex_count = 5;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        component_count = 1;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
};

class isolated_vertices_5_type : public graph_base_data {
public:
    isolated_vertices_5_type() {
        vertex_count = 5;
        edge_count = 0;
        cols_count = 0;
        rows_count = vertex_count + 1;
        component_count = 5;
    }

    std::array<std::int32_t, 5> degrees = { 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 1> cols = { 0 };
    std::array<std::int64_t, 6> rows = { 0, 0, 0, 0, 0, 0 };
};

class three_components_graph_type : public graph_base_data {
public:
    three_components_graph_type() {
        vertex_count = 6;
        edge_count = 4;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        component_count = 3;
    }

    std::array<std::int32_t, 6> degrees = { 2, 2, 2, 1, 1, 0 };
    std::array<std::int32_t, 8> cols = { 1, 2, 0, 2, 0, 1, 4, 3 };
    std::array<std::int64_t, 7> rows = { 0, 2, 4, 6, 7, 8, 8 };
};

class line_graph_5_type : public graph_base_data {
public:
    line_graph_5_type() {
        vertex_count = 5;
        edge_count = 4;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        component_count = 1;
    }

    std::array<std::int32_t, 5> degrees = { 1, 2, 2, 2, 1 };
    std::array<std::int32_t, 8> cols = { 1, 0, 2, 1, 3, 2, 4, 3 };
    std::array<std::int64_t, 6> rows = { 0, 1, 3, 5, 7, 8 };
};

class connected_components_gpu_test {
public:
    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        dal::preview::undirected_adjacency_vector_graph<> g;
        auto& graph_impl = oneapi::dal::detail::get_impl(g);
        auto& vertex_allocator = graph_impl._vertex_allocator;
        auto& edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        std::int32_t* degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t* cols = oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count);
        std::int64_t* rows = oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        std::int32_t* rows_vertex =
            oneapi::dal::preview::detail::allocate(vertex_allocator, rows_count);

        for (int i = 0; i < vertex_count; i++) {
            degrees[i] = graph_data.degrees[i];
        }

        for (int i = 0; i < cols_count; i++) {
            cols[i] = graph_data.cols[i];
        }
        for (int i = 0; i < rows_count; i++) {
            rows[i] = graph_data.rows[i];
            rows_vertex[i] = graph_data.rows[i];
        }
        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        return g;
    }

    sycl::queue get_queue() {
        return sycl::queue{ sycl::default_selector_v };
    }

    template <typename GraphType>
    void check_vertex_partitioning() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t expected_components = graph_data.get_component_count();

        auto q = get_queue();

        std::allocator<char> alloc;
        const auto cc_desc = dal::preview::connected_components::descriptor<
            float,
            dal::preview::connected_components::method::afforest,
            dal::preview::connected_components::task::vertex_partitioning,
            std::allocator<char>>(alloc);

        const auto result = dal::preview::vertex_partitioning(q, cc_desc, g);

        REQUIRE(result.get_component_count() == expected_components);

        const auto labels_table = result.get_labels();
        REQUIRE(labels_table.get_row_count() == vertex_count);

        const auto& labels = static_cast<const dal::homogen_table&>(labels_table);
        const auto labels_data = labels.get_data<std::int32_t>();

        std::set<std::int32_t> unique_labels(labels_data, labels_data + vertex_count);
        REQUIRE(static_cast<std::int64_t>(unique_labels.size()) == expected_components);

        for (std::int64_t u = 0; u < vertex_count; ++u) {
            std::int64_t u_start = graph_data.rows[u];
            std::int64_t u_end = graph_data.rows[u + 1];
            for (std::int64_t i = u_start; i < u_end; ++i) {
                std::int32_t v = graph_data.cols[i];
                REQUIRE(labels_data[u] == labels_data[v]);
            }
        }
    }
};

TEST_M(connected_components_gpu_test, "GPU: triangle graph - 1 component") {
    this->check_vertex_partitioning<triangle_graph_type>();
}

TEST_M(connected_components_gpu_test, "GPU: two components") {
    this->check_vertex_partitioning<two_components_graph_type>();
}

TEST_M(connected_components_gpu_test, "GPU: complete graph K5 - 1 component") {
    this->check_vertex_partitioning<complete_graph_5_type>();
}

TEST_M(connected_components_gpu_test, "GPU: isolated vertices - 5 components") {
    this->check_vertex_partitioning<isolated_vertices_5_type>();
}

TEST_M(connected_components_gpu_test, "GPU: three components") {
    this->check_vertex_partitioning<three_components_graph_type>();
}

TEST_M(connected_components_gpu_test, "GPU: line graph - 1 component") {
    this->check_vertex_partitioning<line_graph_5_type>();
}

TEST_M(connected_components_gpu_test, "GPU: null graph") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    auto q = get_queue();

    std::allocator<char> alloc;
    const auto cc_desc = dal::preview::connected_components::descriptor<
        float,
        dal::preview::connected_components::method::afforest,
        dal::preview::connected_components::task::vertex_partitioning,
        std::allocator<char>>(alloc);

    const auto result = dal::preview::vertex_partitioning(q, cc_desc, null_graph);
    REQUIRE(result.get_component_count() == 0);
}

} // namespace oneapi::dal::algo::connected_components::gpu::test
