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
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/service_functions.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::preview::csv::test {
using namespace oneapi::dal::preview::csv::detail;

class graph_base_data {
protected:
    graph_base_data() = default;

public:
    std::string filename;
    std::vector<std::int32_t> file_content;
    std::vector<std::int32_t> file_content_weighted;

    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::vector<std::int32_t> degrees;
    std::vector<std::int32_t> cols;
    std::vector<std::int32_t> rows;
};

class K10_graph_data : public graph_base_data {
public:
    K10_graph_data() {
        filename = "K10_graph";
        file_content = { 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 1, 2, 1, 3, 1,
                         4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 2, 8,
                         2, 9, 3, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 4, 5, 4, 6, 4, 7, 4, 8, 4,
                         9, 5, 6, 5, 7, 5, 8, 5, 9, 6, 7, 6, 8, 6, 9, 7, 8, 7, 9, 8, 9 };
        file_content_weighted = { 0, 1, 1,  0, 2, 2,  0, 3, 3,  0, 4, 4,  0, 5, 5,  0, 6, 6,
                                  0, 7, 7,  0, 8, 8,  0, 9, 9,  1, 2, 10, 1, 3, 11, 1, 4, 12,
                                  1, 5, 13, 1, 6, 14, 1, 7, 15, 1, 8, 16, 1, 9, 17, 2, 3, 18,
                                  2, 4, 19, 2, 5, 20, 2, 6, 21, 2, 7, 22, 2, 8, 23, 2, 9, 24,
                                  3, 4, 25, 3, 5, 26, 3, 6, 27, 3, 7, 28, 3, 8, 29, 3, 9, 30,
                                  4, 5, 31, 4, 6, 32, 4, 7, 33, 4, 8, 34, 4, 9, 35, 5, 6, 36,
                                  5, 7, 37, 5, 8, 38, 5, 9, 39, 6, 7, 40, 6, 8, 41, 6, 9, 42,
                                  7, 8, 43, 7, 9, 44, 8, 9, 45 };
        vertex_count = 10;
        edge_count = 45;
        degrees = { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
        cols = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 4, 5,
                 6, 7, 8, 9, 0, 1, 2, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 5, 6, 7, 8, 9, 0,
                 1, 2, 3, 4, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 7, 8, 9, 0, 1, 2, 3, 4, 5,
                 6, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        rows = { 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90 };
    }
};

class first_isolated_graph_data : public graph_base_data {
public:
    first_isolated_graph_data() {
        filename = "first_isolated_graph";
        file_content = { 2, 4, 3, 5, 2, 6, 2, 7 }; //vertices 0, 1 are isolated
        file_content_weighted = { 2, 4, 1, 3, 5, 2, 2, 6, 3, 2, 7, 4 };
        vertex_count = 8;
        edge_count = 4;
        degrees = { 0, 0, 3, 1, 1, 1, 1, 1 };
        cols = { 4, 6, 7, 5, 2, 3, 2, 2 };
        rows = { 0, 0, 0, 3, 4, 5, 6, 7, 8 };
    }
};

class middle_isolated_graph_data : public graph_base_data {
public:
    middle_isolated_graph_data() {
        filename = "middle_isolated_graph";
        file_content = { 0, 5, 2, 4, 2, 7 }; //vertices 1, 3, 6 are isolated
        file_content_weighted = { 0, 5, 1, 2, 4, 2, 2, 7, 3 };
        vertex_count = 8;
        edge_count = 3;
        degrees = { 1, 0, 2, 0, 1, 1, 0, 1 };
        cols = { 5, 4, 7, 2, 0, 2 };
        rows = { 0, 1, 1, 3, 3, 4, 5, 5, 6 };
    }
    std::vector<std::int32_t> weights = { 1, 2, 3 };
};

class multiple_edges_graph_data : public graph_base_data {
public:
    multiple_edges_graph_data() {
        filename = "multiple_edges_graph";
        file_content = { 0, 5, 4, 5, 1, 5, 5, 3, 2, 5, 3, 5, 4, 5, 5, 1, 0, 5 };
        file_content_weighted = { 0, 5, 1, 4, 5, 2, 1, 5, 3, 5, 3, 4, 2, 5,
                                  5, 3, 5, 6, 4, 5, 7, 5, 1, 8, 0, 5, 9 };
        vertex_count = 6;
        edge_count = 5;
        degrees = { 1, 1, 1, 1, 1, 5 };
        cols = { 5, 5, 5, 5, 5, 0, 1, 2, 3, 4 };
        rows = { 0, 1, 2, 3, 4, 5, 10 };
    }
};

class self_loops_graph_data : public graph_base_data {
public:
    self_loops_graph_data() {
        filename = "self_loops_graph";
        file_content = { 0, 5, 0, 0, 1, 5, 1, 1, 2, 5, 2, 2, 3, 5, 3, 3, 4, 5, 5, 5 };
        file_content_weighted = { 0, 5, 1, 0, 0, 2, 1, 5, 3, 1, 1, 4, 2, 5, 5,
                                  2, 2, 6, 3, 5, 7, 3, 3, 8, 4, 5, 9, 5, 5, 10 };
        vertex_count = 6;
        edge_count = 5;
        degrees = { 1, 1, 1, 1, 1, 5 };
        cols = { 5, 5, 5, 5, 5, 0, 1, 2, 3, 4 };
        rows = { 0, 1, 2, 3, 4, 5, 10 };
    }
};

class single_edge_graph_data : public graph_base_data {
public:
    single_edge_graph_data() {
        filename = "single_edge_graph";
        file_content = { 0, 9 };
        file_content_weighted = { 0, 9, 1 };
        vertex_count = 10;
        edge_count = 1;
        degrees = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        cols = { 9, 0 };
        rows = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 };
    }
};

class symmetrized_edges_graph_data : public graph_base_data {
public:
    symmetrized_edges_graph_data() {
        filename = "symmetrized_edges_graph";
        file_content = { 0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3, 1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2 };
        file_content_weighted = { 0, 1, 1, 0, 2, 2, 0, 3, 3, 1, 2, 4, 1, 3, 5, 2, 3, 6,
                                  1, 0, 1, 2, 0, 2, 3, 0, 3, 2, 1, 4, 3, 1, 5, 3, 2, 6 };
        vertex_count = 4;
        edge_count = 6;
        degrees = { 3, 3, 3, 3 };
        cols = { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 };
        rows = { 0, 3, 6, 9, 12 };
    }
};

std::int64_t allocated_bytes_count = 0;
bool was_custom_alloc_used = false;

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
        if (!was_custom_alloc_used)
            was_custom_alloc_used = true;
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

class read_graph_test {
public:
    using edge_list_t = typename preview::edge_list<std::int32_t>;
    using weighted_edge_list_t = typename preview::weighted_edge_list<std::int32_t, std::int32_t>;

    using unweighted_graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    using weighted_graph_t =
        dal::preview::undirected_adjacency_vector_graph<std::int32_t, std::int32_t>;

    void write_test_data(graph_base_data& graph_data, bool is_weighted, std::string filename) {
        std::ofstream outf(filename);
        std::int32_t step = 2;
        if (is_weighted) {
            if (outf.is_open()) {
                for (int i = 2; i < graph_data.file_content_weighted.size(); i += 3)
                    outf << graph_data.file_content_weighted[i - 2] << " "
                         << graph_data.file_content_weighted[i - 1] << " "
                         << graph_data.file_content_weighted[i] << std::endl;
            }
            else {
                throw invalid_argument(dal::detail::error_messages::file_not_found());
            }
        }
        else {
            if (outf.is_open()) {
                for (int i = 1; i < graph_data.file_content.size(); i += 2)
                    outf << graph_data.file_content[i - 1] << " " << graph_data.file_content[i]
                         << std::endl;
            }
            else {
                throw invalid_argument(dal::detail::error_messages::file_not_found());
            }
        }
        outf.close();
    }

    void delete_test_data(std::string filename) {
        std::remove(filename.c_str());
    }

    template <typename Graph>
    void check_graph_correctness(graph_base_data& graph_data, const Graph& graph) {
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

    template <typename EdgeList>
    void check_read_edge_list(graph_base_data& graph_data, bool is_weighted, std::string filename) {
        EdgeList elist;
        read_edge_list(filename, elist);
        std::int64_t correct_edge_count = 0;
        std::int32_t step = 0;
        if (is_weighted) {
            correct_edge_count = graph_data.file_content_weighted.size() / 3;
            step = 3;

            REQUIRE(elist.size() == correct_edge_count);
            std::int32_t j = 0;
            std::int32_t num_correct_edges = 0;
            for (int i = 0; i < correct_edge_count; ++i) {
                if (std::get<0>(elist[i]) == graph_data.file_content_weighted[j] &&
                    std::get<1>(elist[i]) == graph_data.file_content_weighted[j + 1])
                    num_correct_edges++;
                j += step;
            }
            REQUIRE(num_correct_edges == correct_edge_count);
        }
        else {
            correct_edge_count = graph_data.file_content.size() / 2;
            step = 2;
            REQUIRE(elist.size() == correct_edge_count);
            std::int32_t j = 0;
            std::int32_t num_correct_edges = 0;
            for (int i = 0; i < correct_edge_count; ++i) {
                if (std::get<0>(elist[i]) == graph_data.file_content[j] &&
                    std::get<1>(elist[i]) == graph_data.file_content[j + 1])
                    num_correct_edges++;
                j += step;
            }
            REQUIRE(num_correct_edges == correct_edge_count);
        }
    }

    void check_convert_to_csr_impl(graph_base_data& graph_data, bool is_weighted) {
        if (is_weighted) {
            weighted_graph_t graph;
            weighted_edge_list_t elist;
            for (int i = 2; i < graph_data.file_content_weighted.size(); i += 3)
                elist.push_back(std::tuple<std::int32_t, std::int32_t, std::int32_t>(
                    graph_data.file_content_weighted[i - 2],
                    graph_data.file_content_weighted[i - 1],
                    graph_data.file_content_weighted[i]));
            convert_to_csr_impl(elist, graph);
            check_graph_correctness(graph_data, graph);
        }
        else {
            weighted_graph_t graph;
            edge_list_t elist;
            for (int i = 1; i < graph_data.file_content.size(); i += 2)
                elist.push_back(
                    std::make_pair(graph_data.file_content[i], graph_data.file_content[i - 1]));
            convert_to_csr_impl(elist, graph);
            check_graph_correctness(graph_data, graph);
        }
    }

    template <typename Graph>
    void check_read_impl(graph_base_data& graph_data, bool is_weighted, std::string filename) {
        if (is_weighted) {
            const auto graph = dal::read<Graph>(dal::csv::data_source{ filename },
                                                dal::preview::read_mode::weighted_edge_list);
            check_graph_correctness(graph_data, graph);
        }
        else {
            const auto graph = dal::read<Graph>(dal::csv::data_source{ filename });
            check_graph_correctness(graph_data, graph);
        }
    }

    void general_check(graph_base_data& graph_data, bool is_weighted, std::string filename) {
        write_test_data(graph_data, is_weighted, filename);
        if (is_weighted) {
            check_read_edge_list<weighted_edge_list_t>(graph_data, is_weighted, filename);
            check_convert_to_csr_impl(graph_data, is_weighted);
            check_read_impl<weighted_graph_t>(graph_data, is_weighted, filename);
        }
        else {
            check_read_edge_list<edge_list_t>(graph_data, is_weighted, filename);
            check_convert_to_csr_impl(graph_data, is_weighted);
            check_read_impl<unweighted_graph_t>(graph_data, is_weighted, filename);
        }
        delete_test_data(filename);
    }
};

#define READ_GRAPH_TEST(name) TEST_M(read_graph_test, name, "[read_graph]")

READ_GRAPH_TEST("K10 edge list, unweighted") {
    K10_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false /*is_weighted =*/, full_filename);
}

READ_GRAPH_TEST("K10 edge list, weighted") {
    K10_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("The first few vertices of the graph are isolated, unweighted") {
    first_isolated_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("The first few vertices of the graph are isolated, weighted") {
    first_isolated_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("Vertices in the middle are isolated, unweighted") {
    middle_isolated_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("Vertices in the middle are isolated, weighted") {
    middle_isolated_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("Edge list with multiple edges, unweighted") {
    multiple_edges_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("Edge list with multiple edges, weighted") {
    multiple_edges_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("Edge list with self-loops, unweighted") {
    self_loops_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("Edge list with self-loops, weighted") {
    self_loops_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("Single edge graph, unweighted") {
    single_edge_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("Single edge graph, weighted") {
    single_edge_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

READ_GRAPH_TEST("Symmetrized edge list, unweighted") {
    symmetrized_edges_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, false, full_filename);
}

READ_GRAPH_TEST("Symmetrized edge list, weighted") {
    symmetrized_edges_graph_data graph_data;
    std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
    this->general_check(graph_data, true, full_filename);
}

// READ_GRAPH_TEST("Check usage of custom allocator") {
//     K10_graph_data graph_data;
//     CountingAllocator<char> alloc;
//     allocated_bytes_count = 0;
//     std::string full_filename = graph_data.filename + std::to_string(std::rand()) + ".csv";
//     this->write_test_data(graph_data, false, full_filename);
//     auto graph =
//             read<unweighted_graph_t>(dal::csv::data_source{ full_filename },
//                                      dal::preview::csv::read_args<unweighted_graph_t>{ alloc });
//     this->delete_test_data(full_filename);
//     REQUIRE(was_custom_alloc_used);
//     REQUIRE(allocated_bytes_count == 0);
// }

} //namespace oneapi::dal::preview::csv::test
