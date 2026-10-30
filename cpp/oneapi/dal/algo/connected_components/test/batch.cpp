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
#include <algorithm>
#include <queue>

#include "oneapi/dal/algo/connected_components/vertex_partitioning.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/graph/service_functions.hpp"

namespace oneapi::dal::algo::connected_components::test {
namespace dal = oneapi::dal;

class graph_base_data {
protected:
    graph_base_data() = default;
    graph_base_data(std::int64_t vertex_count,
                    std::int64_t edge_count,
                    std::int64_t components_count)
            : vertex_count(vertex_count),
              edge_count(edge_count),
              cols_count(edge_count * 2),
              rows_count(vertex_count + 1),
              components_count(components_count),
              degrees(vertex_count),
              cols(edge_count * 2),
              rows(vertex_count + 1) {}

public:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
    std::int64_t components_count;

    std::vector<std::int32_t> degrees;
    std::vector<std::int32_t> cols;
    std::vector<std::int64_t> rows;
};

void adj_list_to_csr(std::vector<std::vector<std::int32_t>>& g,
                     std::vector<std::int32_t>& degrees,
                     std::vector<std::int32_t>& cols,
                     std::vector<std::int64_t>& rows) {
    assert(g.size() == degrees.size());
    assert(g.size() + 1 == rows.size());
    std::int64_t vertex_count = g.size();
    for (std::int64_t v = 0, index = 0; v < vertex_count; ++v) {
        std::sort(g[v].begin(), g[v].end());
        degrees[v] = g[v].size();
        rows[v + 1] = rows[v] + degrees[v];
        for (std::int32_t u : g[v]) {
            cols[index++] = u;
        }
    }
}

class null_graph_data : public graph_base_data {
public:
    null_graph_data() : graph_base_data(0, 0, 0) {}
};

class single_vertices_data : public graph_base_data {
public:
    single_vertices_data(std::int64_t vertex_count = 1)
            : graph_base_data(vertex_count, 0, vertex_count) {
        assert(vertex_count >= 1);
    }
};

class complete_graph_data : public graph_base_data {
public:
    complete_graph_data(std::int64_t vertex_count)
            : graph_base_data(vertex_count, vertex_count * (vertex_count - 1) / 2, 1) {
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
        for (std::int64_t index = 1; index < vertex_count; ++index) {
            rows[index] = rows[index - 1] + vertex_count - 1;
        }
    }
};

class lolipop_graph_data : public graph_base_data {
public:
    lolipop_graph_data(std::int64_t head_vertex_count, std::int64_t tail_vertex_count)
            : graph_base_data(head_vertex_count + tail_vertex_count,
                              head_vertex_count * (head_vertex_count - 1) / 2 + tail_vertex_count,
                              1) {
        assert(head_vertex_count >= 1);
        assert(tail_vertex_count >= 1);
        std::fill(degrees.begin(), degrees.begin() + head_vertex_count, head_vertex_count - 1);
        std::fill(degrees.begin() + head_vertex_count, degrees.end(), 2);
        degrees[head_vertex_count - 1] = head_vertex_count;
        degrees.back() = 1;
        std::int64_t cols_index = 0;
        for (std::int64_t vertex_index = 0; vertex_index < head_vertex_count; ++vertex_index) {
            for (std::int64_t neighbour = 0; neighbour < head_vertex_count; ++neighbour) {
                if (neighbour != vertex_index) {
                    cols[cols_index++] = neighbour;
                }
            }
        }
        cols[cols_index++] = head_vertex_count;
        for (std::int64_t vertex_index = 0; vertex_index < tail_vertex_count - 1; ++vertex_index) {
            cols[cols_index++] = head_vertex_count + vertex_index - 1;
            cols[cols_index++] = head_vertex_count + vertex_index + 1;
        }
        cols[cols_index] = head_vertex_count + tail_vertex_count - 2;
        for (std::int64_t index = 0; index < head_vertex_count + tail_vertex_count; ++index) {
            rows[index + 1] = rows[index] + degrees[index];
        }
    }
};

class line_graph_data : public graph_base_data {
public:
    line_graph_data(std::int64_t vertex_count)
            : graph_base_data(vertex_count, vertex_count - 1, 1) {
        assert(vertex_count >= 1);
        degrees.front() = degrees.back() = 1;
        std::fill(degrees.begin() + 1, degrees.end() - 1, 2);
        for (std::int64_t v = 0; v < vertex_count; ++v) {
            rows[v + 1] = rows[v] + degrees[v];
        }
        for (std::int64_t v = 0, index = 0; v < vertex_count; v++) {
            if (v != 0) {
                cols[index++] = v - 1;
            }
            if (v != vertex_count - 1) {
                cols[index++] = v + 1;
            }
        }
    }
};

class reindexed_line_17_graph_data : public graph_base_data {
public:
    reindexed_line_17_graph_data() {
        vertex_count = 17;
        edge_count = 16;
        cols_count = 32;
        rows_count = 18;
        components_count = 1;

        degrees = { { 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1 } };
        cols = { 13, 9, 11, 15, 12, 14, 10, 16, 12, 13, 10, 15, 9, 16, 11, 14,
                 1,  7, 4,  6,  2,  8,  3,  5,  0,  5,  3,  8,  2, 6,  4,  7 };
        rows = { 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32 };
    }
};

class reindexed_binary_tree_graph_data : public graph_base_data {
public:
    reindexed_binary_tree_graph_data(std::int64_t depth)
            : graph_base_data((1 << depth) - 1, (1 << depth) - 2, 1) {
        assert(depth >= 1);
        std::vector<std::int32_t> order(vertex_count);
        for (std::int64_t v = 0; v < vertex_count; ++v) {
            order[v] = vertex_count - v - 1;
        }
        for (std::int64_t cur_depth = 0; cur_depth < depth; ++cur_depth) {
            std::int64_t size = 1ll << cur_depth;
            std::int64_t index = size - 1;
            if (cur_depth % 2 == 0) {
                std::reverse(order.begin() + index, order.begin() + index + size);
            }
        }
        std::vector<std::vector<std::int32_t>> g(vertex_count);
        for (std::int64_t index = 0; index < vertex_count / 2; ++index) {
            std::int32_t left_index = index * 2 + 1;
            std::int32_t right_index = index * 2 + 2;
            std::int32_t v = order[index];
            std::int32_t v_left = order[left_index];
            std::int32_t v_right = order[right_index];
            g[v].push_back(v_left);
            g[v].push_back(v_right);
            g[v_left].push_back(v);
            g[v_right].push_back(v);
        }
        adj_list_to_csr(g, degrees, cols, rows);
    }
};

class grid_graph_data : public graph_base_data {
public:
    grid_graph_data(std::int64_t n, std::int64_t m)
            : graph_base_data(n * m, n * (m - 1) + (n - 1) * m, 1) {
        assert(n >= 1);
        assert(m >= 1);
        std::vector<std::vector<std::int32_t>> g(n * m);
        for (std::int64_t row = 0; row < n; ++row) {
            for (std::int64_t col = 0; col < m; ++col) {
                std::int64_t v = row * m + col;
                if (row + 1 < n) {
                    std::int64_t u = v + m;
                    g[v].push_back(u);
                    g[u].push_back(v);
                }
                if (col + 1 < m) {
                    std::int64_t u = v + 1;
                    g[v].push_back(u);
                    g[u].push_back(v);
                }
            }
        }
        adj_list_to_csr(g, degrees, cols, rows);
    }
};

class star_graph_data : public graph_base_data {
public:
    star_graph_data(std::int64_t leaves) : graph_base_data(leaves + 1, leaves, 1) {
        assert(leaves >= 0);
        degrees[0] = leaves;
        std::fill(degrees.begin() + 1, degrees.end(), 1);
        for (std::int64_t v = 0; v < leaves; ++v) {
            cols[v] = v + 1;
            rows[v + 1] = rows[v] + degrees[v];
        }
    }
};

class combined_graph_data : public graph_base_data {
public:
    combined_graph_data() : graph_base_data(0, 0, 0) {}

    void add_graph(const graph_base_data& g) {
        std::int64_t vertex_shift = vertex_count;
        vertex_count += g.vertex_count;
        edge_count += g.edge_count;
        cols_count += g.cols_count;
        rows_count += g.vertex_count;
        components_count += g.components_count;

        degrees.reserve(vertex_count);
        cols.reserve(cols_count);
        rows.reserve(rows_count);
        for (std::int64_t v = 0; v < g.vertex_count; ++v) {
            degrees.push_back(g.degrees[v]);
            rows.push_back(rows.back() + g.degrees[v]);
        }
        for (std::int64_t index = 0; index < g.cols_count; ++index) {
            cols.push_back(g.cols[index] + vertex_shift);
        }
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

class connected_components_test {
public:
    using graph_type = dal::preview::undirected_adjacency_vector_graph<std::int32_t>;

    graph_type create_graph(const graph_base_data& graph_data) {
        graph_type graph;

        auto& graph_impl = oneapi::dal::detail::get_impl(graph);
        auto& vertex_allocator = graph_impl._vertex_allocator;
        auto& edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.vertex_count;
        const std::int64_t edge_count = graph_data.edge_count;
        const std::int64_t cols_count = graph_data.cols_count;
        const std::int64_t rows_count = graph_data.rows_count;

        typedef std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>
            int32_traits_t;
        typedef std::allocator_traits<std::allocator<char>>::rebind_traits<std::int64_t>
            int64_traits_t;
        std::int32_t* degrees = int32_traits_t::allocate(vertex_allocator, vertex_count);
        std::int32_t* cols = int32_traits_t::allocate(vertex_allocator, cols_count);
        std::int64_t* rows = int64_traits_t::allocate(edge_allocator, rows_count);
        std::int32_t* rows_vertex = int32_traits_t::allocate(vertex_allocator, rows_count);

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
        return graph;
    }

    void check_labels_correctness(const graph_base_data& graph_data,
                                  const oneapi::dal::table& table) {
        REQUIRE(table.get_row_count() == graph_data.vertex_count);
        if (!table.has_data())
            return;
        REQUIRE(table.get_column_count() == 1);

        auto table_data = oneapi::dal::row_accessor<const std::int32_t>(table).pull();
        const auto result_labels = table_data.get_data();

        std::vector<std::int32_t> labels(graph_data.vertex_count, -1);
        std::queue<std::int32_t> q;
        std::vector<std::int32_t> unique_labels;
        for (std::int64_t v = 0; v < graph_data.vertex_count; ++v) {
            if (labels[v] != -1) {
                continue;
            }
            std::int32_t label_value = result_labels[v];
            unique_labels.push_back(label_value);
            labels[v] = label_value;
            q.push(v);
            while (!q.empty()) {
                std::int32_t v_in = q.front();
                q.pop();
                for (std::int64_t index = graph_data.rows[v_in]; index < graph_data.rows[v_in + 1];
                     ++index) {
                    std::int32_t v_to = graph_data.cols[index];
                    if (labels[v_to] == -1) {
                        labels[v_to] = label_value;
                        q.push(v_to);
                    }
                    else if (labels[v_to] != label_value) {
                        throw std::runtime_error(
                            "Incorrect graph structure, graph contains directed edges");
                    }
                }
            }
        }
        std::sort(unique_labels.begin(), unique_labels.end());
        REQUIRE(dal::detail::integral_cast<std::int64_t>(unique_labels.size()) ==
                graph_data.components_count);

        bool correctly_reindexed = unique_labels[0] == 0;
        for (std::int64_t index = 0; index < graph_data.components_count; ++index) {
            correctly_reindexed &= index == unique_labels[index];
        }
        REQUIRE(correctly_reindexed);

        bool corrently_labeled = true;
        for (std::int64_t v = 0; v < graph_data.vertex_count; ++v) {
            corrently_labeled &= labels[v] == result_labels[v];
        }
        REQUIRE(corrently_labeled);
    }

    void check_connected_components(const graph_base_data& graph_data) {
        const graph_type graph = create_graph(graph_data);
        allocated_bytes_count = 0;
        {
            CountingAllocator<char> alloc;
            const auto desc = dal::preview::connected_components::descriptor<
                float,
                oneapi::dal::preview::connected_components::method::afforest,
                oneapi::dal::preview::connected_components::task::vertex_partitioning,
                CountingAllocator<char>>(alloc);
            const auto result = dal::preview::vertex_partitioning(desc, graph);

            REQUIRE(graph_data.components_count == result.get_component_count());
            REQUIRE_NOTHROW(check_labels_correctness(graph_data, result.get_labels()));
        }
        REQUIRE(allocated_bytes_count == 0);
    }
};

#define CONNECTED_COMPONENTS_TEST(name) \
    TEST_M(connected_components_test, name, "[connected_components]")

CONNECTED_COMPONENTS_TEST("Empty graph case") {
    null_graph_data graph_data;
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST("Isolated vertices case") {
    single_vertices_data graph_data(2048);
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST("Check correctness: Grid-10-15 graph") {
    grid_graph_data graph_data(10, 15);
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST("Check correctness: Star-100 graph") {
    star_graph_data graph_data(100);
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST("Check correctness: K8 graph + K6 graph + K5 graph") {
    combined_graph_data graph_data;
    graph_data.add_graph(complete_graph_data(8));
    graph_data.add_graph(complete_graph_data(6));
    graph_data.add_graph(complete_graph_data(5));
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST(
    "Check correctness: Lolipop-1000-500 graph + 300 K3 graphs + 100 Single vertices graphs") {
    combined_graph_data graph_data;
    graph_data.add_graph(lolipop_graph_data(1000, 500));
    for (std::int32_t i = 0; i < 300; ++i) {
        graph_data.add_graph(complete_graph_data(3));
    }
    graph_data.add_graph(single_vertices_data(100));
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST("Check correctness: Line-1025 graph") {
    line_graph_data graph_data(1025);
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST(
    "Check correctness of the compress + order_component_ids: Reindexed-Line-17 graph") {
    reindexed_line_17_graph_data graph_data;
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST(
    "Check correctness of the compress + order_component_ids: 1025 Reindexed-Line-17 graphs") {
    combined_graph_data graph_data;
    for (std::int32_t i = 0; i < 1025; ++i) {
        graph_data.add_graph(reindexed_line_17_graph_data());
    }
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST(
    "Check correctness of link for get_vertex_degree(u) >= neighbors_round: Reindexed-Binary-tree-11 graph") {
    reindexed_binary_tree_graph_data graph_data(11);
    this->check_connected_components(graph_data);
}

CONNECTED_COMPONENTS_TEST(
    "Check correctness of link for get_vertex_degree(u) >= neighbors_round: 1025 Reindexed-Binary-tree-4 graphs") {
    combined_graph_data graph_data;
    for (std::int32_t i = 0; i < 1025; ++i) {
        graph_data.add_graph(reindexed_binary_tree_graph_data(4));
    }
    this->check_connected_components(graph_data);
}

} // namespace oneapi::dal::algo::connected_components::test
