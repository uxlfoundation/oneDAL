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

using namespace Catch;

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

//Stochastic block model graph containing 5 communities with 4 vertices in each.
//Probability of an edge appearing between different communities = 0.02, within a community - 0.7.
class sbm_graph_data : public graph_base_data {
public:
    sbm_graph_data() : graph_base_data(20, 24) {
        cols = { 1,  2,  3,  16, 0,  2,  3, 4,  16, 0,  1,  0,  1,  1,  6,  7,
                 4,  7,  5,  6,  9,  10, 8, 11, 8,  11, 9,  10, 13, 14, 12, 14,
                 15, 12, 13, 15, 13, 14, 0, 1,  17, 18, 19, 16, 16, 19, 16, 18 };
        rows = { 0, 4, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28, 30, 33, 36, 38, 43, 44, 46, 48 };
        degrees = { 4, 5, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 5, 1, 2, 2 };
    }
};

//Edge weights within the community are several times greater than edge weights between the communities.
//Graph also contains isolated vertices.
class combined_graph_data : public graph_base_data {
public:
    combined_graph_data() : graph_base_data(15, 12) {
        rows = { 0, 1, 4, 5, 7, 7, 8, 10, 12, 13, 14, 16, 16, 20, 20, 24 };
        cols = { 14, 3, 7, 12, 7, 1, 8, 14, 10, 12, 1, 2, 3, 14, 6, 12, 1, 6, 10, 14, 0, 5, 9, 12 };
        degrees = { 1, 3, 1, 2, 0, 1, 2, 2, 1, 1, 2, 0, 4, 0, 4 };
    }
};

//Two complete graphs connected by one edge.
class barbell_graph_data : public graph_base_data {
public:
    barbell_graph_data() : graph_base_data(20, 91) {
        rows = { 0,   9,   18,  27,  36,  45,  54,  63,  72,  81, 91,
                 101, 110, 119, 128, 137, 146, 155, 164, 173, 182 };
        cols = {
            1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  2,  3,  4,  5,  6,  7,  8,  9,  0,  1,  3,
            4,  5,  6,  7,  8,  9,  0,  1,  2,  4,  5,  6,  7,  8,  9,  0,  1,  2,  3,  5,  6,
            7,  8,  9,  0,  1,  2,  3,  4,  6,  7,  8,  9,  0,  1,  2,  3,  4,  5,  7,  8,  9,
            0,  1,  2,  3,  4,  5,  6,  8,  9,  0,  1,  2,  3,  4,  5,  6,  7,  9,  0,  1,  2,
            3,  4,  5,  6,  7,  8,  10, 9,  11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 12, 13, 14,
            15, 16, 17, 18, 19, 10, 11, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 14, 15, 16, 17,
            18, 19, 10, 11, 12, 13, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 16, 17, 18, 19, 10,
            11, 12, 13, 14, 15, 17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 18, 19, 10, 11, 12, 13,
            14, 15, 16, 17, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        };

        degrees = { 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
    }
};

class two_complete_graphs_data : public graph_base_data {
public:
    two_complete_graphs_data() : graph_base_data(10, 20) {
        std::fill(degrees.begin(), degrees.end(), 4);
        cols = { 1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3,
                 6, 7, 8, 9, 5, 7, 8, 9, 5, 6, 8, 9, 5, 6, 7, 9, 5, 6, 7, 8 };
        rows = { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40 };
    }
};

class complete_graph_data : public graph_base_data {
public:
    complete_graph_data(std::int64_t vertex_count)
            : graph_base_data(vertex_count, vertex_count * (vertex_count - 1) / 2) {
        assert(vertex_count >= 1);
        std::fill(degrees.begin(), degrees.end(), vertex_count - 1);
        for (std::int64_t vertex_index = 0, cols_index = 0; vertex_index < vertex_count;
             ++vertex_index) {
            for (std::int64_t neighbour = 0; neighbour < vertex_count; ++neighbour) {
                if (neighbour != vertex_index) {
                    cols[cols_index++] = neighbour;
                }
            }
        }
        rows[0] = 0;
        for (std::int64_t index = 1; index <= vertex_count; ++index) {
            rows[index] = rows[index - 1] + vertex_count - 1;
        }
    }
};

class single_vertex_data : public graph_base_data {
public:
    single_vertex_data(std::int64_t vertex_count = 1) : graph_base_data(vertex_count, 0) {
        assert(vertex_count >= 1);
    }
};

class karate_club_graph_data : public graph_base_data {
public:
    karate_club_graph_data() : graph_base_data(34, 78) {
        cols = { 1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,
                 13, 17, 19, 21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12,
                 13, 0,  6,  10, 0,  6,  10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32,
                 33, 2,  33, 0,  4,  5,  0,  0,  3,  0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,
                 0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32, 33, 25, 27, 29, 32, 33, 25, 27,
                 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26, 32, 33, 1,  8,  32,
                 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8,
                 9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32 };
        rows = { 0,  16, 25, 35, 41, 44, 48, 52,  56,  61,  63,  66,  67,  69,  74,  76,  78, 80,
                 82, 84, 87, 89, 91, 93, 98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156 };
        degrees = { 16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2,  2,
                    2,  2, 3,  2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17 };
    }
};

class rand_generated_graph_data : public graph_base_data {
public:
    rand_generated_graph_data() : graph_base_data(20, 35) {
        cols = { 2,  17, 3,  4,  5,  10, 14, 0,  8,  18, 19, 1,  4,  6, 16, 1,  3, 10,
                 16, 1,  9,  15, 17, 3,  10, 19, 17, 19, 2,  12, 13, 5, 16, 1,  4, 6,
                 18, 14, 16, 17, 19, 8,  14, 8,  19, 1,  11, 12, 18, 5, 16, 18, 3, 4,
                 9,  11, 15, 0,  5,  7,  11, 2,  10, 14, 15, 2,  6,  7, 11, 13 };
        rows = { 0, 2, 7, 11, 15, 19, 23, 26, 28, 31, 33, 37, 41, 43, 45, 49, 52, 57, 61, 65, 70 };
        degrees = { 2, 5, 4, 4, 4, 4, 3, 2, 3, 2, 4, 4, 2, 2, 4, 3, 5, 4, 4, 5 };
    }
};

std::int64_t allocated_bytes_count = 0;

template <class T>
struct CountingAllocator {
    typedef T value_type;
    typedef T* pointer;

    CountingAllocator() {}

    template <class U>
    CountingAllocator(const CountingAllocator<U>& other) {}

    template <class U>
    auto operator=(const CountingAllocator<U>& other) {
        return *this;
    }

    template <class U>
    bool operator!=(const CountingAllocator<U>& other) {
        return true;
    }

    bool operator!=(const CountingAllocator<T>& other) {
        return false;
    }

    T* allocate(const std::size_t n) {
        allocated_bytes_count += n * sizeof(T);

        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(pv);
    }

    void deallocate(T* const p, std::size_t n) noexcept {
        allocated_bytes_count -= n * sizeof(T);
        free(p);
    }
};

template <typename EdgeValueType>
struct GraphTypeInfo;

template <>
struct GraphTypeInfo<double> {
    using graph_type =
        typename dal::preview::undirected_adjacency_vector_graph<std::int32_t, double>;
};

template <>
struct GraphTypeInfo<std::int32_t> {
    using graph_type =
        typename dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t>;
};

class louvain_test {
public:
    static bool is_nan(double x) {
        return x != x;
    }

    template <typename EdgeValueType>
    auto create_graph(const graph_base_data& graph_data, std::vector<EdgeValueType>& weights) {
        using graph_type = typename GraphTypeInfo<EdgeValueType>::graph_type;

        graph_type graph;

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
        EdgeValueType* edge_weights =
            oneapi::dal::preview::detail::allocate(edge_user_value_allocator, 2 * edge_count);

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

        for (int i = 0; i < 2 * edge_count; i++) {
            edge_weights[i] = weights[i];
        }

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        graph_impl.set_edge_values(edge_weights, edge_count);
        return graph;
    }

    void check_result_correctness(const graph_base_data& graph_data,
                                  const dal::preview::louvain::vertex_partitioning_result<>& result,
                                  std::vector<std::int32_t>& expected_labels,
                                  std::int64_t expected_community_count) {
        UNSCOPED_INFO("Modularity is NaN");
        REQUIRE(!is_nan(result.get_modularity()));
        UNSCOPED_INFO("Modularity value is not in [-1/2; 1]");
        REQUIRE(Approx(result.get_modularity()) <= 1.0);
        UNSCOPED_INFO("Modularity value is not in [-1/2; 1]");
        REQUIRE(Approx(result.get_modularity()) >= -0.5);

        UNSCOPED_INFO("Community count is incorrect");
        REQUIRE(result.get_community_count() == expected_community_count);

        const auto result_table = result.get_labels();
        REQUIRE(result_table.get_row_count() == graph_data.vertex_count);
        if (!result_table.has_data())
            return;
        REQUIRE(result_table.get_column_count() == 1);
        auto table_data = oneapi::dal::row_accessor<const std::int32_t>(result_table).pull();
        const auto result_labels = table_data.get_data();
        std::int64_t correct_labels_count = 0;
        for (int u = 0; u < graph_data.vertex_count; ++u) {
            if (result_labels[u] == expected_labels[u])
                correct_labels_count++;
        }
        UNSCOPED_INFO("Labels are incorrect");
        REQUIRE(correct_labels_count == graph_data.vertex_count);
    }

    template <typename EdgeValueType>
    void check_louvain(const graph_base_data& graph_data,
                       std::vector<EdgeValueType>& weights,
                       std::vector<std::int32_t>& expected_labels,
                       std::int64_t expected_community_count) {
        const auto graph = create_graph<EdgeValueType>(graph_data, weights);
        const auto louvain_desc = dal::preview::louvain::descriptor<>();
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
        check_result_correctness(graph_data, result, expected_labels, expected_community_count);
    }

    template <typename EdgeValueType>
    void check_louvain(const graph_base_data& graph_data,
                       std::vector<EdgeValueType>& weights,
                       const table& init_partition,
                       std::vector<std::int32_t>& expected_labels,
                       std::int64_t expected_community_count) {
        const auto graph = create_graph<EdgeValueType>(graph_data, weights);
        const auto louvain_desc = dal::preview::louvain::descriptor<>();
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph, init_partition);
        check_result_correctness(graph_data, result, expected_labels, expected_community_count);
    }

    template <typename EdgeValueType>
    void check_resolution_values(const graph_base_data& graph_data,
                                 std::vector<EdgeValueType>& weights,
                                 double resolution,
                                 bool is_modularity_valid) {
        const auto graph = create_graph<EdgeValueType>(graph_data, weights);
        const auto louvain_desc = dal::preview::louvain::descriptor<>()
                                      .set_resolution(resolution)
                                      .set_accuracy_threshold(0.0001)
                                      .set_max_iteration_count(0);
        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
        if (is_modularity_valid) {
            UNSCOPED_INFO("Modularity is NaN");
            REQUIRE(!is_nan(result.get_modularity()));
            UNSCOPED_INFO("Modularity value is not in [-1/2; 1]");
            REQUIRE(Approx(result.get_modularity()) <= 1.0);
            UNSCOPED_INFO("Modularity value is not in [-1/2; 1]");
            REQUIRE(Approx(result.get_modularity()) >= -0.5);
        }
        const auto result_table = result.get_labels();
        REQUIRE(result_table.get_row_count() == graph_data.vertex_count);
        REQUIRE(result_table.has_data());
        REQUIRE(result_table.get_column_count() == 1);
    }
};

#define LOUVAIN_TEST(name) TEST_M(louvain_test, name, "[louvain]")

LOUVAIN_TEST("Null input graph") {
    const auto louvain_desc = dal::preview::louvain::descriptor<>();
    dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t> graph;
    const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    REQUIRE(Approx(result.get_modularity()) == 0.0);
    REQUIRE(result.get_community_count() == 0);
    REQUIRE(!result.get_labels().has_data());
}

LOUVAIN_TEST("Null input graph with empty initial partition table") {
    dal::homogen_table::table initial_labels;
    const auto louvain_desc = dal::preview::louvain::descriptor<>();
    dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t> graph;
    const auto result = dal::preview::vertex_partitioning(louvain_desc, graph, initial_labels);
    REQUIRE(Approx(result.get_modularity()) == 0.0);
    REQUIRE(result.get_community_count() == 0);
    REQUIRE(!result.get_labels().has_data());
}

LOUVAIN_TEST("K5 graph + K5 graph, edge weights are nonzero and equal") {
    two_complete_graphs_data graph_data;
    std::vector<std::int32_t> expected_labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
    std::int64_t expected_community_count = 2;
    SECTION("Int32 weights") {
        std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
        std::fill(int_weights.begin(), int_weights.end(), 1);
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights(2 * graph_data.edge_count);
        std::fill(double_weights.begin(), double_weights.end(), 1.5);
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

//Fails - community_count and labels are correct, but modularity is NaN
// LOUVAIN_TEST("K5 graph + K5 graph, zero weights") {
//     two_complete_graphs_data graph_data;
//     std::vector<std::int32_t> expected_labels = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//     std::int64_t expected_community_count = 10;
//     SECTION("Int32 weights"){
//         std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
//         std::fill(int_weights.begin(), int_weights.end(), 0);
//         this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
//     }
//     SECTION("Double weights"){
//         std::vector<double> double_weights(2 * graph_data.edge_count);
//         std::fill(double_weights.begin(), double_weights.end(), -5);
//         this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
//     }
// }

LOUVAIN_TEST("Barbell graph, edge weights are nonzero and equal") {
    barbell_graph_data graph_data;
    std::vector<std::int32_t> expected_labels = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::int64_t expected_community_count = 2;
    SECTION("Int32 weights") {
        std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
        std::fill(int_weights.begin(), int_weights.end(), 10);
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights(2 * graph_data.edge_count);
        std::fill(double_weights.begin(), double_weights.end(), 0.5);
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

LOUVAIN_TEST("Combined graph") {
    combined_graph_data graph_data;
    std::int64_t expected_community_count = 6;
    std::vector<std::int32_t> expected_labels = { 0, 1, 1, 1, 2, 0, 3, 1, 1, 0, 3, 4, 3, 5, 0 };
    SECTION("Int32 weights") {
        std::vector<std::int32_t> int_weights = { 100, 12, 14, 3,  17,  12,  10,  150,
                                                  60,  70, 14, 17, 10,  120, 60,  50,
                                                  3,   70, 50, 1,  100, 150, 120, 1 };
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights = { 100.5, 12.3, 14.7, 3.0,  17.8,  12.4,  10.2,  150.1,
                                               60.5,  70.8, 14.9, 17.8, 10.9,  120.1, 60.2,  50.1,
                                               3.1,   70.4, 50.3, 1.0,  100.2, 150.1, 120.2, 1.0 };
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

LOUVAIN_TEST("K_20 graph, edge weights are nonzero and equal") {
    complete_graph_data graph_data(20);
    std::vector<std::int32_t> expected_labels(graph_data.vertex_count);
    std::fill(expected_labels.begin(), expected_labels.end(), 0);
    std::int64_t expected_community_count = 1;
    SECTION("Int32_t weights") {
        std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
        std::fill(int_weights.begin(), int_weights.end(), 5);
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights(2 * graph_data.edge_count);
        std::fill(double_weights.begin(), double_weights.end(), 0.5);
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

LOUVAIN_TEST("SBM graph, edge weights are nonzero and equal") {
    sbm_graph_data graph_data;
    std::vector<std::int32_t> expected_labels = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
                                                  2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
    std::int64_t expected_community_count = 5;
    SECTION("Int32 weights") {
        std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
        std::fill(int_weights.begin(), int_weights.end(), 10);
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights(2 * graph_data.edge_count);
        std::fill(double_weights.begin(), double_weights.end(), 10.0);
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

LOUVAIN_TEST("Random generated graph with 20 vertices and 35 edges, int32 weights") {
    rand_generated_graph_data graph_data;
    std::vector<std::int32_t> int_weights = {
        12,  122, 707, 922, 633, 762, 764, 12,  640, 291, 799, 707, 748, 620, 454, 922, 748, 398,
        565, 633, 386, 147, 943, 620, 391, 959, 319, 917, 640, 949, 871, 386, 661, 762, 398, 391,
        31,  239, 683, 898, 806, 949, 52,  871, 104, 764, 239, 52,  625, 147, 198, 917, 454, 565,
        661, 683, 198, 122, 943, 319, 898, 291, 31,  625, 917, 799, 959, 917, 806, 104
    };
    std::vector<std::int32_t> expected_labels = { 0, 1, 2, 1, 1, 0, 2, 2, 3, 0,
                                                  1, 0, 3, 3, 4, 4, 0, 0, 4, 2 };
    std::int64_t expected_community_count = 5;
    this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
}

LOUVAIN_TEST("Random generated graph with 20 vertices and 35 edges, double weights") {
    rand_generated_graph_data graph_data;
    std::vector<double> double_weights = {
        844.31, 393.1,  28.56,  680.05, 67.83,  635.11, 749.47, 844.31, 554.88, 527.54,
        94.38,  28.56,  178.51, 169.21, 479.49, 680.05, 178.51, 318.85, 604.71, 67.83,
        860.01, 135.33, 640.29, 169.21, 720.99, 182.1,  505.99, 126.71, 554.88, 107.91,
        757.97, 860.01, 955.43, 635.11, 318.85, 720.99, 688,    396.95, 329.42, 226.37,
        775.34, 107.91, 756.78, 757.97, 724.05, 749.47, 396.95, 756.78, 614.4,  135.33,
        313.37, 799.64, 479.49, 604.71, 955.43, 329.42, 313.37, 393.1,  640.29, 505.99,
        226.37, 527.54, 688,    614.4,  799.64, 94.38,  182.1,  126.71, 775.34, 724.05
    };
    std::vector<std::int32_t> expected_labels = { 0, 1, 0, 2, 1, 2, 1, 0, 3, 2,
                                                  1, 3, 4, 3, 4, 4, 2, 0, 4, 3 };
    std::int64_t expected_community_count = 5;
    this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
}

LOUVAIN_TEST("Zachary's karate club graph, edge weights are nonzero and equal") {
    karate_club_graph_data graph_data;
    std::int64_t expected_community_count = 4;
    std::vector<std::int32_t> expected_labels = {
        0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 1,
        0, 2, 0, 2, 0, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2
    };
    SECTION("Int32 weights") {
        std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
        std::fill(int_weights.begin(), int_weights.end(), 10);
        this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
    }
    SECTION("Double weights") {
        std::vector<double> double_weights(2 * graph_data.edge_count);
        std::fill(double_weights.begin(), double_weights.end(), 0.5);
        this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
    }
}

LOUVAIN_TEST("SBM graph with different initial partitions") {
    sbm_graph_data graph_data;
    std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
    std::fill(int_weights.begin(), int_weights.end(), 10);
    std::vector<std::int32_t> expected_labels = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
                                                  2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
    std::int64_t expected_community_count = 5;
    const std::int64_t data_first[] = { 0,  17, 4, 8,  1,  13, 9,  7,  7,  9,
                                        10, 11, 1, 18, 12, 16, 13, 12, 14, 1 };
    const std::int64_t data_second[] = { 4, 19, 10, 6,  0,  8,  12, 16, 3, 9,
                                         2, 7,  1,  18, 19, 11, 5,  19, 8, 15 };
    const auto initial_labels_first = dal::homogen_table::wrap(data_first, 20, 1);
    const auto initial_labels_second = dal::homogen_table::wrap(data_second, 20, 1);
    this->check_louvain(graph_data,
                        int_weights,
                        initial_labels_first,
                        expected_labels,
                        expected_community_count);
    this->check_louvain(graph_data,
                        int_weights,
                        initial_labels_second,
                        expected_labels,
                        expected_community_count);
}

LOUVAIN_TEST("Random generated graph with different initial partitions") {
    rand_generated_graph_data graph_data;
    std::vector<std::int32_t> int_weights = {
        12,  122, 707, 922, 633, 762, 764, 12,  640, 291, 799, 707, 748, 620, 454, 922, 748, 398,
        565, 633, 386, 147, 943, 620, 391, 959, 319, 917, 640, 949, 871, 386, 661, 762, 398, 391,
        31,  239, 683, 898, 806, 949, 52,  871, 104, 764, 239, 52,  625, 147, 198, 917, 454, 565,
        661, 683, 198, 122, 943, 319, 898, 291, 31,  625, 917, 799, 959, 917, 806, 104
    };
    std::vector<std::int32_t> expected_labels = { 0, 1, 2, 1, 1, 0, 2, 2, 3, 0,
                                                  1, 0, 3, 3, 4, 4, 0, 0, 4, 2 };
    std::int64_t expected_community_count = 5;
    const std::int64_t data_first[] = { 0,  17, 4, 8,  1,  13, 9,  7,  7,  9,
                                        10, 11, 1, 18, 12, 16, 13, 12, 14, 1 };
    const std::int64_t data_second[] = { 4, 19, 10, 6,  0,  8,  12, 16, 3, 9,
                                         2, 7,  1,  18, 19, 11, 5,  19, 8, 15 };
    const auto initial_labels_first = dal::homogen_table::wrap(data_first, 20, 1);
    const auto initial_labels_second = dal::homogen_table::wrap(data_second, 20, 1);
    this->check_louvain(graph_data,
                        int_weights,
                        initial_labels_first,
                        expected_labels,
                        expected_community_count);
    this->check_louvain(graph_data,
                        int_weights,
                        initial_labels_second,
                        expected_labels,
                        expected_community_count);
}

//Fails - modularity is NaN, community_count and labels are incorrect
// LOUVAIN_TEST("K_20, all weights = int32_max") {
//     complete_graph_data graph_data(20);
//     std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
//     std::fill(int_weights.begin(), int_weights.end(), std::numeric_limits<std::int32_t>::max());
//     std::vector<std::int32_t> expected_labels(graph_data.vertex_count);
//     std::fill(expected_labels.begin(), expected_labels.end(), 0);
//     std::int64_t expected_community_count = 1;
//     this->check_louvain(graph_data, int_weights, expected_labels, expected_community_count);
// }

//Fails - modularity is NaN, community_count and labels are incorrect
// LOUVAIN_TEST("K_20, all weights = double_max") {
//     complete_graph_data graph_data(20);
//     std::vector<double> double_weights(2 * graph_data.edge_count);
//     std::fill(double_weights.begin(), double_weights.end(), std::numeric_limits<double>::max());
//     std::vector<std::int32_t> expected_labels(graph_data.vertex_count);
//     std::fill(expected_labels.begin(), expected_labels.end(), 0);
//     std::int64_t expected_community_count = 1;
//     this->check_louvain(graph_data, double_weights, expected_labels, expected_community_count);
// }

LOUVAIN_TEST("Counting allocator test, null graph") {
    dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t> graph;
    allocated_bytes_count = 0;
    CountingAllocator<char> alloc;
    const auto louvain_desc =
        dal::preview::louvain::descriptor<float,
                                          oneapi::dal::preview::louvain::method::by_default,
                                          oneapi::dal::preview::louvain::task::by_default,
                                          CountingAllocator<char>>(alloc);
    const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    REQUIRE(allocated_bytes_count == 0);
}

LOUVAIN_TEST("Counting allocator test, SBM graph") {
    sbm_graph_data graph_data;
    std::vector<double> double_weights(2 * graph_data.edge_count);
    std::fill(double_weights.begin(), double_weights.end(), 0.5);
    std::vector<std::int32_t> expected_labels = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
                                                  2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
    const auto graph = create_graph<double>(graph_data, double_weights);
    allocated_bytes_count = 0;
    CountingAllocator<char> alloc;
    const auto louvain_desc =
        dal::preview::louvain::descriptor<float,
                                          oneapi::dal::preview::louvain::method::by_default,
                                          oneapi::dal::preview::louvain::task::by_default,
                                          CountingAllocator<char>>(alloc);
    const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    REQUIRE(allocated_bytes_count == 0);
}

LOUVAIN_TEST("Counting allocator test, K20 graph") {
    complete_graph_data graph_data(20);
    std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
    std::fill(int_weights.begin(), int_weights.end(), 10);
    std::vector<std::int32_t> expected_labels(graph_data.vertex_count);
    std::fill(expected_labels.begin(), expected_labels.end(), 0);
    const auto graph = create_graph<std::int32_t>(graph_data, int_weights);
    allocated_bytes_count = 0;
    CountingAllocator<char> alloc;
    const auto louvain_desc =
        dal::preview::louvain::descriptor<float,
                                          oneapi::dal::preview::louvain::method::by_default,
                                          oneapi::dal::preview::louvain::task::by_default,
                                          CountingAllocator<char>>(alloc);
    const auto result = dal::preview::vertex_partitioning(louvain_desc, graph);
    REQUIRE(allocated_bytes_count == 0);
}

LOUVAIN_TEST("Different resolution values, SBM graph") {
    sbm_graph_data graph_data;
    std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
    std::fill(int_weights.begin(), int_weights.end(), 10);
    this->check_resolution_values(graph_data, int_weights, 0, true /*is_modularity_valid=*/);
    this->check_resolution_values(graph_data, int_weights, 9.3, true);
    this->check_resolution_values(graph_data, int_weights, 9.4, false);
}

LOUVAIN_TEST("Different resolution values, K20 graph") {
    complete_graph_data graph_data(20);
    std::vector<std::int32_t> int_weights(2 * graph_data.edge_count);
    std::fill(int_weights.begin(), int_weights.end(), 10);
    this->check_resolution_values(graph_data, int_weights, 0, true /*is_modularity_valid=*/);
    this->check_resolution_values(graph_data, int_weights, 1, true);
    this->check_resolution_values(graph_data, int_weights, 10, true);
    this->check_resolution_values(graph_data, int_weights, 10.1, false);
}

} //namespace oneapi::dal::algo::louvain::test
