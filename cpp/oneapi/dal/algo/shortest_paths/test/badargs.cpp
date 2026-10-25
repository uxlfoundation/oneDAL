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

#include <array>

#include "oneapi/dal/algo/shortest_paths/traverse.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::shortest_paths::test {

namespace dal = oneapi::dal;

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
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
};

class example_graph_type : public graph_base_data {
public:
    example_graph_type() {
        vertex_count = 6;
        edge_count = 9;
        cols_count = 9;
        rows_count = 7;
    }
    std::array<std::int64_t, 7> rows = { 0, 2, 4, 6, 8, 9, 9 };
    std::array<std::int32_t, 9> cols = { 1, 2, 3, 4, 3, 4, 4, 5, 5 };
    std::array<double, 9> edge_weights = { 10, 20, 50, 10, 20, 33, 20, 2, 1 };
};

class empty_graph_type : public graph_base_data {
public:
    empty_graph_type() {
        vertex_count = 0;
        edge_count = 0;
        cols_count = 0;
        rows_count = 1;
    }
    std::array<std::int64_t, 1> rows = { 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<double, 0> edge_weights = {};
};

class negative_weights_graph_type : public graph_base_data {
public:
    negative_weights_graph_type() {
        vertex_count = 3;
        edge_count = 2;
        cols_count = 2;
        rows_count = 4;
    }
    std::array<std::int64_t, 4> rows = { 0, 1, 2, 2 };
    std::array<std::int32_t, 2> cols = { 1, 2 };
    std::array<double, 2> edge_weights = { -10, -20 };
};

class shortest_paths_badargs_test {
public:
    template <typename GraphType>
    void check_shortest_paths(double delta, std::int64_t source, bool nothing_to_compute = false) {
        using namespace dal::preview::shortest_paths;
        GraphType graph_data;

        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            int,
            double,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights.data());

        const auto& graph = graph_builder.get_graph();

        auto result_type = nothing_to_compute
                               ? optional_results::distances & optional_results::predecessors
                               : optional_results::distances | optional_results::predecessors;

        const auto shortest_paths_desc = descriptor<>(source, delta, result_type);

        const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, graph);
    }
};

#define SHORTEST_PATHS_BADARG_TEST(name) \
    TEST_M(shortest_paths_badargs_test, name, "[shortest_paths][badarg]")

SHORTEST_PATHS_BADARG_TEST("Check delta is > 0") {
    REQUIRE_THROWS_AS((this->check_shortest_paths<example_graph_type>(-0.001, 1)),
                      invalid_argument);
    REQUIRE_THROWS_AS((this->check_shortest_paths<example_graph_type>(-10000, 2)),
                      invalid_argument);
}

SHORTEST_PATHS_BADARG_TEST("Check source is in graph ") {
    REQUIRE_THROWS_AS((this->check_shortest_paths<example_graph_type>(5, -2)), invalid_argument);
    REQUIRE_THROWS_AS((this->check_shortest_paths<example_graph_type>(5, 100)), invalid_argument);
}

SHORTEST_PATHS_BADARG_TEST("Check empty graph") {
    REQUIRE_THROWS_AS((this->check_shortest_paths<empty_graph_type>(5, 0)), invalid_argument);
}

// SHORTEST_PATHS_BADARG_TEST("Check edges are non-negative") {
//     REQUIRE_THROWS_AS((this->check_shortest_paths<negative_weights_graph_type>(5, 0)),
//                       invalid_argument);
// }

SHORTEST_PATHS_BADARG_TEST("Check nothing to calculate case") {
    REQUIRE_THROWS_AS((this->check_shortest_paths<example_graph_type>(5, 0, true)),
                      invalid_argument);
}
} // namespace oneapi::dal::algo::shortest_paths::test
