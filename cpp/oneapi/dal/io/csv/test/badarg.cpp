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

#include <fstream>
#include <vector>

#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/service_functions.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::preview::csv::test {
using namespace oneapi::dal::preview::csv::detail;

class graph_base_data {
protected:
    graph_base_data() = default;

public:
    std::int64_t vertex_count;
    std::int64_t edge_count;

    std::vector<std::int32_t> degrees;
    std::vector<std::int32_t> cols;
    std::vector<std::int64_t> rows;
};

class K4_graph_data : public graph_base_data {
public:
    K4_graph_data() {
        vertex_count = 4;
        edge_count = 6;
        degrees = { 3, 3, 3, 3 };
        cols = { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 };
        rows = { 0, 3, 6, 9, 12 };
    }
};

class read_graph_badarg_test {
public:
    using unweighted_graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    using weighted_graph_t =
        dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t>;

    void write_test_data(std::string filename, std::string file_content) {
        std::ofstream outf(filename);
        if (outf.is_open()) {
            std::string delimiter = ",";
            std::size_t pos = 0;
            while ((pos = file_content.find(delimiter)) != std::string::npos) {
                outf << file_content.substr(0, pos) << std::endl;
                file_content.erase(0, pos + delimiter.length());
            }
            outf << file_content << std::endl;
        }
        else {
            throw invalid_argument(dal::detail::error_messages::file_not_found());
        }
        outf.close();
    }

    void delete_test_data(std::string filename) {
        std::remove(filename.c_str());
    }

    template <typename GraphType, typename Graph>
    void check_graph_correctness(const Graph& graph) {
        GraphType graph_data;

        REQUIRE(dal::preview::get_vertex_count(graph) == graph_data.vertex_count);
        REQUIRE(dal::preview::get_edge_count(graph) == graph_data.edge_count);

        auto& graph_impl = oneapi::dal::detail::get_impl(graph);
        const auto& t = graph_impl.get_topology();

        std::int32_t correct_elements_count = 0;
        REQUIRE(t._degrees.get_count() == graph_data.vertex_count);
        for (int i = 0; i < graph_data.vertex_count; ++i) {
            if (t._degrees[i] == graph_data.degrees[i])
                correct_elements_count++;
        }
        REQUIRE(correct_elements_count == graph_data.vertex_count);

        correct_elements_count = 0;
        REQUIRE(t._rows.get_count() == graph_data.vertex_count + 1);
        REQUIRE(t._rows_vertex.get_count() == graph_data.vertex_count + 1);
        for (int i = 0; i < graph_data.vertex_count + 1; ++i) {
            if (t._rows[i] == graph_data.rows[i] && t._rows_vertex[i] == graph_data.rows[i])
                correct_elements_count++;
        }
        REQUIRE(correct_elements_count == graph_data.vertex_count + 1);

        correct_elements_count = 0;
        REQUIRE(t._cols.get_count() == 2 * graph_data.edge_count);
        for (int i = 0; i < graph_data.edge_count * 2; ++i) {
            if (t._cols[i] == graph_data.cols[i])
                correct_elements_count++;
        }
        REQUIRE(correct_elements_count == 2 * graph_data.edge_count);
    }
};

#define READ_GRAPH_BADARG_TEST(name) TEST_M(read_graph_badarg_test, name, "[read_graph][badarg]")

READ_GRAPH_BADARG_TEST("Empty input file") {
    std::string file_content = "";
    std::string filename = "empty_file.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Non-exist input file") {
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ "non_exist_file.csv" }),
                      invalid_argument);
}

READ_GRAPH_BADARG_TEST("Edge list contains non-numeric characters, unweighted graph") {
    std::string filename = "non_numeric_chars_unweighted.csv";
    std::string file_content = "0 1,0 2,0 3,ab1 2,1 3,2 3";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1,0 2,0 3,1ab 2,1 3,2 3";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1,0 2,0 3,1 ab2,1 3,2 3";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1,0 2,0 3,1 2ab,1 3,2 3";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Edge list contains non-numeric characters, weighted graph") {
    std::string filename = "non_numeric_chars_weighted.csv";
    std::string file_content = "ab0 1 1,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0ab 1 1,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 ab1 1,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1ab 1,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1 ab1,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);

    file_content = "0 1 1ab,0 2 2,0 3 3,1 2 4";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Extra values in line, unweighted graph") {
    std::string file_content = "0 1 10,0 2,1 2,3 4,5 6";
    std::string filename = "extra_values_unweighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Extra values in line, weighted graph") {
    std::string file_content = "0 1 3 10,0 2 1,1 2 3,3 4 6,5 6 7";
    std::string filename = "extra_values_weighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Edge is incomplete, unweighted graph") {
    std::string file_content = "0 1,0 2,1,3 4,5 6";
    std::string filename = "incomplete_edge_unweighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Edge is incomplete, weighted graph") {
    std::string file_content = "0 1 3,0 2 1,1,3 4 6,5 6 7";
    std::string filename = "incomplete_edge_weighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Not all edges have weights for weighted graph") {
    std::string file_content = "0 1 1,0 2 2,0 3,1 2 4,1 3,2 3 6";
    std::string filename = "not_all_weights.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Weights column is missing for weighted graph") {
    std::string file_content = "0 1,0 2,0 3,1 2,1 3,2 3";
    std::string filename = "missed_weights.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Vertex indices is not integer, unweighted") {
    std::string file_content = "0 1.1,0.2 2.2,0 3.3,1.4 2.4,1 3.5,2.6 3.6";
    std::string filename = "not_int_ids_unweighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Vertex indices is not integer, weighted graph") {
    std::string file_content = "0 1.1 1,0.2 2.2 2,0 3.3 3,1.4 2.4 4,1 3.5 5,2.6 3.6 6";
    std::string filename = "not_int_ids_weighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Negative vertex indices, unweighted graph") {
    std::string file_content = "0 1,0 2,0 -3,1 2,1 3,-2 3";
    std::string filename = "negative_ids_unweighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Negative vertex indices, weighted graph") {
    std::string file_content = "0 1 1,0 2 2,0 -3 3,1 2 4,1 3 5,-2 3 6";
    std::string filename = "negative_ids_weighted.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<weighted_graph_t>(dal::csv::data_source{ filename },
                                                  dal::preview::read_mode::weighted_edge_list),
                      invalid_argument);
    delete_test_data(filename);
}

READ_GRAPH_BADARG_TEST("Vertex id is greater than int32_max") {
    std::string file_content = "0 1,0 2,0 2147483648,1 2,1 3,2 3";
    std::string filename = "vertex_id_overflow.csv";
    write_test_data(filename, file_content);
    REQUIRE_THROWS_AS(dal::read<unweighted_graph_t>(dal::csv::data_source{ filename }),
                      range_error);
    delete_test_data(filename);
}

} //namespace oneapi::dal::preview::csv::test
