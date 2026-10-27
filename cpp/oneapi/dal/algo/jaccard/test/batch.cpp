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

#include <array>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::jaccard::test {

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

class complete_graph_33_type : public graph_base_data {
public:
    complete_graph_33_type() {
        vertex_count = 33;
        edge_count = 528;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 33> degrees = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                             32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                             32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
    std::array<std::int32_t, 1056> cols = {
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 0,  1,  2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,
        1,  2,  3,  4,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,
        6,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,
        2,  3,  4,  5,  6,  7,  8,  9,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,
        5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28,
        29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 0,
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 29, 30, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 0,  1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 31, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    std::array<std::int64_t, 34> rows = { 0,   32,  64,  96,  128, 160,  192, 224, 256,
                                          288, 320, 352, 384, 416, 448,  480, 512, 544,
                                          576, 608, 640, 672, 704, 736,  768, 800, 832,
                                          864, 896, 928, 960, 992, 1024, 1056 };
};

class zero_jaccard_coeff_graph_type : public graph_base_data {
public:
    zero_jaccard_coeff_graph_type() {
        vertex_count = 34;
        edge_count = 17;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }
    std::array<std::int32_t, 34> degrees = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    std::array<std::int32_t, 34> cols = { 1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11, 10,
                                          13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                                          25, 24, 27, 26, 29, 28, 31, 30, 33, 32 };
    std::array<std::int64_t, 35> rows = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34 };
};

class jaccard_test {
public:
    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        dal::preview::undirected_adjacency_vector_graph<> g;
        auto &graph_impl = oneapi::dal::detail::get_impl(g);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        std::int32_t *degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t *cols = oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count);
        std::int64_t *rows = oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        std::int32_t *rows_vertex =
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

    template <typename GraphType,
              typename Task,
              typename VertexPairsDataType,
              typename JaccardCoeffsDataType>
    void check_jaccard(const oneapi::dal::preview::jaccard::detail::descriptor_base<Task> &desc,
                       std::int64_t correct_nonzero_coeff_count,
                       const VertexPairsDataType &correct_vertex_pairs,
                       const JaccardCoeffsDataType &correct_jaccard_coeffs) {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        dal::preview::jaccard::caching_builder builder;
        const auto result_vertex_similarity = dal::preview::vertex_similarity(desc, g, builder);

        UNSCOPED_INFO("The number of non-zero jaccard coefficients was determined incorrectly");
        const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
        REQUIRE(nonzero_coeff_count == correct_nonzero_coeff_count);

        UNSCOPED_INFO("Pairs of vertices with non-zero jaccard coefficient were found wrong");
        auto vertex_pairs_table = result_vertex_similarity.get_vertex_pairs();
        homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
        const auto vertex_pairs_data = vertex_pairs.get_data<int>();
        std::int64_t correct_pair_count = 0;
        std::int64_t element_count = (desc.get_row_range_end() - desc.get_row_range_begin()) *
                                     (desc.get_column_range_end() - desc.get_column_range_begin());
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (vertex_pairs_data[i] == correct_vertex_pairs[i] &&
                vertex_pairs_data[i + element_count] == correct_vertex_pairs[i + element_count])
                correct_pair_count++;
        }
        REQUIRE(correct_pair_count == nonzero_coeff_count);

        UNSCOPED_INFO("Jaccard coefficients are not correct");
        auto coeffs_table = result_vertex_similarity.get_coeffs();
        homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
        const auto jaccard_coeffs_data = coeffs.get_data<float>();
        int correct_coeff_count = 0;
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (Catch::Approx(jaccard_coeffs_data[i]) == correct_jaccard_coeffs[i])
                correct_coeff_count++;
        }
        REQUIRE(correct_coeff_count == nonzero_coeff_count);
    }

    template <typename Graph, typename Task>
    void check_jaccard_zero_coeffs_only(
        const oneapi::dal::preview::jaccard::detail::descriptor_base<Task> &desc,
        const Graph &g) {
        dal::preview::jaccard::caching_builder builder;
        const auto result_vertex_similarity = dal::preview::vertex_similarity(desc, g, builder);

        UNSCOPED_INFO("The number of non-zero jaccard coefficients != 0");
        const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
        REQUIRE(nonzero_coeff_count == 0);
    }
};

// TEST_M(jaccard_test,
//        "Complete graph, row of <16 elements, left of the diagonal, without diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 0, 10 });
//     std::array<std::int64_t, 20> vertex_pairs = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//                                                   0,  1,  2,  3,  4,  5,  6,  7,  8,  9 };
//     std::array<float, 10> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 10, vertex_pairs, jaccard_coeffs);
// }

TEST_M(jaccard_test,
       "Complete graph, row of <16 elements, left of the diagonal, contains diagonal element") {
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 23, 33 });
    std::array<std::int64_t, 20> vertex_pairs = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
    std::array<float, 10> jaccard_coeffs;
    jaccard_coeffs.fill(0.93939);
    jaccard_coeffs[9] = 1.0;
    this->check_jaccard<complete_graph_33_type>(jaccard_desc, 10, vertex_pairs, jaccard_coeffs);
}

// TEST_M(
//     jaccard_test,
//     "Complete graph, row of >16 and <32 elements, left of the diagonal, without diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 0, 25 });
//     std::array<std::int64_t, 50> vertex_pairs = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//                                                   32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//                                                   32, 32, 32, 32, 32, 0,  1,  2,  3,  4,
//                                                   5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
//                                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
//     std::array<float, 25> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 25, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(
//     jaccard_test,
//     "Complete graph, row of >16 and <32 elements, left of the diagonal, contains diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 8, 33 });
//     std::array<std::int64_t, 50> vertex_pairs = { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//                                                   32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//                                                   32, 32, 32, 32, 32, 8,  9,  10, 11, 12,
//                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
//                                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
//     std::array<float, 25> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     jaccard_coeffs[24] = 1.0;
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 25, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(jaccard_test,
//        "Complete graph, row of 32 elements, left of the diagonal, without diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 0, 32 });
//     std::array<std::int64_t, 64> vertex_pairs = {
//         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
//         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
//     };
//     std::array<float, 32> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 32, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(jaccard_test,
//        "Complete graph, row of 32 elements, left of the diagonal, contains diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 32, 33 }, { 1, 33 });
//     std::array<std::int64_t, 64> vertex_pairs = {
//         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
//         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
//         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
//     };
//     std::array<float, 32> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     jaccard_coeffs[31] = 1.0;
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 32, vertex_pairs, jaccard_coeffs);
// }

TEST_M(jaccard_test,
       "Complete graph, row of <16 elements, right of the diagonal, without diagonal element") {
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 11 });
    std::array<std::int64_t, 20> vertex_pairs = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::array<float, 10> jaccard_coeffs;
    jaccard_coeffs.fill(0.93939);
    this->check_jaccard<complete_graph_33_type>(jaccard_desc, 10, vertex_pairs, jaccard_coeffs);
}

TEST_M(jaccard_test,
       "Complete graph, row of <16 elements, right of the diagonal, contains diagonal element") {
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 0, 10 });
    std::array<std::int64_t, 20> vertex_pairs = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::array<float, 10> jaccard_coeffs;
    jaccard_coeffs.fill(0.93939);
    jaccard_coeffs[0] = 1.0;
    this->check_jaccard<complete_graph_33_type>(jaccard_desc, 10, vertex_pairs, jaccard_coeffs);
}

// TEST_M(
//     jaccard_test,
//     "Complete graph, row of >16 and <32 elements, right of the diagonal, without diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 26 });
//     std::array<std::int64_t, 50> vertex_pairs = {
//         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
//         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
//     };
//     std::array<float, 25> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 25, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(
//     jaccard_test,
//     "Complete graph, row of >16 and <32 elements, right of the diagonal, contains diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 0, 25 });
//     std::array<std::int64_t, 50> vertex_pairs = {
//         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
//     };
//     std::array<float, 25> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     jaccard_coeffs[0] = 1.0;
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 25, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(jaccard_test,
//        "Complete graph, row of 32 elements, right of the diagonal, without diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 33 });
//     std::array<std::int64_t, 64> vertex_pairs = {
//         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
//         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
//         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
//     };
//     std::array<float, 33> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 32, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(jaccard_test,
//        "Complete graph, row of 32 elements, right of the diagonal, contains diagonal element") {
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 0, 32 });
//     std::array<std::int64_t, 64> vertex_pairs = {
//         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,
//         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
//         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
//     };
//     std::array<float, 33> jaccard_coeffs;
//     jaccard_coeffs.fill(0.93939);
//     jaccard_coeffs[0] = 1.0;
//     this->check_jaccard<complete_graph_33_type>(jaccard_desc, 32, vertex_pairs, jaccard_coeffs);
// }

// TEST_M(jaccard_test, "Zero jaccard coeffs graph, row of <16 elements, left of the diagonal") {
//     const auto g = create_graph<zero_jaccard_coeff_graph_type>();
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 33, 34 }, { 0, 10 });
//     this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
// }

// TEST_M(jaccard_test,
//        "Zero jaccard coeffs graph, row of >16 and <32 elements, left of the diagonal") {
//     const auto g = create_graph<zero_jaccard_coeff_graph_type>();
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 33, 34 }, { 0, 25 });
//     this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
// }

// TEST_M(jaccard_test, "Zero jaccard coeffs graph, row of 32 elements, left of the diagonal") {
//     const auto g = create_graph<zero_jaccard_coeff_graph_type>();
//     auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 33, 34 }, { 0, 32 });
//     this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
// }

TEST_M(jaccard_test, "Zero jaccard coeffs graph, row of <16 elements, right of the diagonal") {
    const auto g = create_graph<zero_jaccard_coeff_graph_type>();
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 10 });
    this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
}

TEST_M(jaccard_test,
       "Zero jaccard coeffs graph, row of >16 and <32 elements, right of the diagonal") {
    const auto g = create_graph<zero_jaccard_coeff_graph_type>();
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 25 });
    this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
}

TEST_M(jaccard_test, "Zero jaccard coeffs graph, row of 32 elements, right of the diagonal") {
    const auto g = create_graph<zero_jaccard_coeff_graph_type>();
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 1 }, { 1, 34 });
    this->check_jaccard_zero_coeffs_only<>(jaccard_desc, g);
}

TEST_M(jaccard_test, "Null graph") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 0 }, { 0, 0 });
    dal::preview::jaccard::caching_builder builder;
    const auto result_vertex_similarity =
        dal::preview::vertex_similarity(jaccard_desc, null_graph, builder);

    UNSCOPED_INFO("The number of non-zero jaccard coefficients != 0");
    const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
    REQUIRE(nonzero_coeff_count == 0);

    UNSCOPED_INFO("The non-empty table of vertex pairs was returned");
    auto vertex_pairs_table = result_vertex_similarity.get_vertex_pairs();
    homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
    REQUIRE(vertex_pairs.has_data() == false);

    UNSCOPED_INFO("The non-empty table of Jaccard coefficients was returned");
    auto coeffs_table = result_vertex_similarity.get_coeffs();
    homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
    REQUIRE(coeffs.has_data() == false);
}

} // namespace oneapi::dal::algo::jaccard::test
