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

#include <vector>

#include "oneapi/dal/algo/louvain/vertex_partitioning.hpp"
#include "oneapi/dal/graph/service_functions.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::louvain::test {

namespace dal = oneapi::dal;

class graph_base_data {
protected:
    graph_base_data() = default;
    graph_base_data(std::int64_t vertex_count, std::int64_t edge_count)
            : vertex_count(vertex_count),
              edge_count(edge_count),
              cols_count(edge_count * 2),
              rows_count(vertex_count + 1),
              degrees(vertex_count),
              cols(edge_count * 2),
              rows(vertex_count + 1) {}

public:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;

    std::vector<std::int32_t> degrees;
    std::vector<std::int32_t> cols;
    std::vector<std::int64_t> rows;
};

class complete_graph_data : public graph_base_data {
public:
    complete_graph_data(std::int64_t vertex_count)
            : graph_base_data(vertex_count, vertex_count * (vertex_count - 1) / 2) {
        assert(vertex_count >= 1);
        std::fill(degrees.begin(), degrees.end(), static_cast<std::int32_t>(vertex_count - 1));
        for (std::int64_t vertex_index = 0, cols_index = 0; vertex_index < vertex_count;
             ++vertex_index) {
            for (std::int64_t neighbour = 0; neighbour < vertex_count; ++neighbour) {
                if (neighbour != vertex_index) {
                    cols[cols_index++] = neighbour;
                }
            }
        }
        for (std::int64_t index = 1; index < vertex_count; ++index) {
            rows[index] = rows[index - 1] + vertex_count - 1;
        }
    }
};

class louvain_badarg_test {
public:
    using graph_t = dal::preview::undirected_adjacency_vector_graph<std::int32_t, double>;

    graph_t create_graph(const graph_base_data& graph_data) {
        graph_t graph;

        auto& graph_impl = oneapi::dal::detail::get_impl(graph);

        auto& vertex_allocator = graph_impl._vertex_allocator;
        auto& edge_allocator = graph_impl._edge_allocator;
        auto& edge_user_value_allocator = graph_impl._edge_user_value_allocator;

        const std::int64_t vertex_count = graph_data.vertex_count;
        const std::int64_t edge_count = graph_data.edge_count;
        const std::int64_t cols_count = graph_data.cols_count;
        const std::int64_t rows_count = graph_data.rows_count;

        std::int32_t* degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t* cols = oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count);
        std::int64_t* rows = oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        std::int32_t* rows_vertex =
            oneapi::dal::preview::detail::allocate(vertex_allocator, rows_count);
        double* edge_weights =
            oneapi::dal::preview::detail::allocate(edge_user_value_allocator, edge_count);

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

        for (int i = 0; i < edge_count; i++) {
            edge_weights[i] = 1;
        }

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        graph_impl.set_edge_values(edge_weights, edge_count);
        return graph;
    }

    void check_accuracy_threshold(const graph_base_data& graph_data, double accuracy_threshold) {
        const auto graph = create_graph(graph_data);
        const auto louvain_desc = dal::preview::louvain::descriptor<>()
                                      .set_resolution(1)
                                      .set_accuracy_threshold(accuracy_threshold)
                                      .set_max_iteration_count(0);
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    }

    void check_resolution(const graph_base_data& graph_data, double resolution) {
        const auto graph = create_graph(graph_data);
        const auto louvain_desc = dal::preview::louvain::descriptor<>()
                                      .set_resolution(resolution)
                                      .set_accuracy_threshold(0.0001)
                                      .set_max_iteration_count(0);
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    }

    void check_max_iteration_count(const graph_base_data& graph_data, double max_iteration_count) {
        const auto graph = create_graph(graph_data);
        const auto louvain_desc = dal::preview::louvain::descriptor<>()
                                      .set_resolution(1)
                                      .set_accuracy_threshold(0.0001)
                                      .set_max_iteration_count(max_iteration_count);
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    }

    void check_initial_partition(const graph_base_data& graph_data, const table& init_partition) {
        const auto graph = create_graph(graph_data);
        const auto louvain_desc = dal::preview::louvain::descriptor<>();
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph, init_partition);
    }
};

#define LOUVAIN_BADARG_TEST(name) TEST_M(louvain_badarg_test, name, "[louvain][badarg]")

LOUVAIN_BADARG_TEST("Negative accuracy_threshold") {
    double accuracy_threshold = -2.5;
    REQUIRE_THROWS_AS(this->check_accuracy_threshold(complete_graph_data(5), accuracy_threshold),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Negative resolution") {
    double resolution = -1.5;
    REQUIRE_THROWS_AS(this->check_resolution(complete_graph_data(5), resolution), invalid_argument);
}

LOUVAIN_BADARG_TEST("Negative max_iteration_count") {
    std::int64_t max_iteration_count = -15;
    REQUIRE_THROWS_AS(this->check_max_iteration_count(complete_graph_data(5), max_iteration_count),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Initial partition size less than vertex_count") {
    const std::int64_t data[] = { 0 };
    const auto initial_labels = dal::homogen_table::wrap(data, 1, 1);
    REQUIRE_THROWS_AS(this->check_initial_partition(complete_graph_data(5), initial_labels),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Initial partition size greater than vertex_count") {
    const std::int64_t data[] = { 0, 0, 0, 1, 1, 1 };
    const auto initial_labels = dal::homogen_table::wrap(data, 6, 1);
    REQUIRE_THROWS_AS(this->check_initial_partition(complete_graph_data(5), initial_labels),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Invalid layout of initial partition") {
    const std::int64_t data[] = { 0, 0, 0, 0, 1, 2, 3, 4 };
    const auto initial_labels = dal::homogen_table::wrap(data, 4, 2);
    REQUIRE_THROWS_AS(this->check_initial_partition(complete_graph_data(8), initial_labels),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Negative values in initial  partition") {
    const std::int64_t data[] = { 0, -1, -2, -3, -4 };
    const auto initial_labels = dal::homogen_table::wrap(data, 5, 1);
    REQUIRE_THROWS_AS(this->check_initial_partition(complete_graph_data(5), initial_labels),
                      invalid_argument);
}

LOUVAIN_BADARG_TEST("Community labels >= vertex count") {
    const std::int64_t data[] = { 0, 5, 5, 5, 6 };
    const auto initial_labels = dal::homogen_table::wrap(data, 5, 1);
    REQUIRE_THROWS_AS(this->check_initial_partition(complete_graph_data(5), initial_labels),
                      invalid_argument);
}

} //namespace oneapi::dal::algo::louvain::test
