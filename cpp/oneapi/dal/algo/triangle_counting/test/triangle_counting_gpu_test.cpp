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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/triangle_counting/vertex_ranking.hpp"
#include "oneapi/dal/graph/detail/device_csr_topology.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::triangle_counting::gpu::test {

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_global_triangle_count() const {
        return global_triangle_count;
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
    std::int64_t global_triangle_count;
};

class complete_graph_5_type : public graph_base_data {
public:
    complete_graph_5_type() {
        vertex_count = 5;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 10;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
    std::array<std::int64_t, 5> local_triangles = { 6, 6, 6, 6, 6 };
};

class complete_graph_9_type : public graph_base_data {
public:
    complete_graph_9_type() {
        vertex_count = 9;
        edge_count = 36;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 84;
    }

    std::array<std::int32_t, 9> degrees = { 8, 8, 8, 8, 8, 8, 8, 8, 8 };
    std::array<std::int32_t, 72> cols = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 4, 5, 6, 7, 8, 0, 1,
                                          3, 4, 5, 6, 7, 8, 0, 1, 2, 4, 5, 6, 7, 8, 0, 1, 2, 3,
                                          5, 6, 7, 8, 0, 1, 2, 3, 4, 6, 7, 8, 0, 1, 2, 3, 4, 5,
                                          7, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int64_t, 10> rows = { 0, 8, 16, 24, 32, 40, 48, 56, 64, 72 };
    std::array<std::int64_t, 9> local_triangles = { 28, 28, 28, 28, 28, 28, 28, 28, 28 };
};

class triangle_graph_type : public graph_base_data {
public:
    triangle_graph_type() {
        vertex_count = 3;
        edge_count = 3;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 1;
    }

    std::array<std::int32_t, 3> degrees = { 2, 2, 2 };
    std::array<std::int32_t, 6> cols = { 1, 2, 0, 2, 0, 1 };
    std::array<std::int64_t, 4> rows = { 0, 2, 4, 6 };
    std::array<std::int64_t, 3> local_triangles = { 1, 1, 1 };
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        vertex_count = 2;
        edge_count = 1;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 0;
    }

    std::array<std::int32_t, 2> degrees = { 1, 1 };
    std::array<std::int32_t, 2> cols = { 1, 0 };
    std::array<std::int64_t, 3> rows = { 0, 1, 2 };
    std::array<std::int64_t, 2> local_triangles = { 0, 0 };
};

class wheel_graph_6_type : public graph_base_data {
public:
    wheel_graph_6_type() {
        vertex_count = 6;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 5;
    }

    std::array<std::int32_t, 6> degrees = { 5, 3, 3, 3, 3, 3 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 5, 0, 2, 5, 0, 1, 3, 0, 2, 4, 0, 3, 5, 0, 1, 4
    };
    std::array<std::int64_t, 7> rows = { 0, 5, 8, 11, 14, 17, 20 };
    std::array<std::int64_t, 6> local_triangles = { 5, 2, 2, 2, 2, 2 };
};

class graph_with_isolated_vertices_10_type : public graph_base_data {
public:
    graph_with_isolated_vertices_10_type() {
        vertex_count = 10;
        edge_count = 11;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
        global_triangle_count = 5;
    }

    std::array<std::int32_t, 10> degrees = { 5, 3, 2, 0, 3, 4, 0, 2, 0, 3 };
    std::array<std::int32_t, 22> cols = { 1, 2, 4, 5, 7, 0, 5, 9, 0, 7, 0,
                                          5, 9, 0, 1, 4, 9, 0, 2, 1, 4, 5 };
    std::array<std::int64_t, 11> rows = { 0, 5, 8, 10, 10, 13, 17, 17, 19, 19, 22 };
    std::array<std::int64_t, 10> local_triangles = { 3, 2, 1, 0, 2, 4, 0, 1, 0, 2 };
};

class triangle_counting_gpu_test {
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
    void check_local_task() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();

        auto q = get_queue();

        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::local,
            std::allocator<char>>(alloc);

        const auto result = dal::preview::vertex_ranking(q, tc_desc, g);

        const auto local_triangles_table = result.get_ranks();
        const auto& local_triangles =
            static_cast<const dal::homogen_table&>(local_triangles_table);
        const auto local_triangles_data = local_triangles.get_data<std::int64_t>();

        REQUIRE(local_triangles_table.get_row_count() == vertex_count);

        int correct_local_triangle_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (local_triangles_data[i] == graph_data.local_triangles[i]) {
                correct_local_triangle_count++;
            }
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_local_and_global_task() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t global_triangle_count = graph_data.get_global_triangle_count();

        auto q = get_queue();

        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::local_and_global,
            std::allocator<char>>(alloc);

        const auto result = dal::preview::vertex_ranking(q, tc_desc, g);

        const auto local_triangles_table = result.get_ranks();
        const auto& local_triangles =
            static_cast<const dal::homogen_table&>(local_triangles_table);
        const auto local_triangles_data = local_triangles.get_data<std::int64_t>();

        REQUIRE(result.get_global_rank() == global_triangle_count);
        REQUIRE(local_triangles_table.get_row_count() == vertex_count);

        int correct_local_triangle_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (local_triangles_data[i] == graph_data.local_triangles[i]) {
                correct_local_triangle_count++;
            }
        }
        REQUIRE(correct_local_triangle_count == vertex_count);
    }

    template <typename GraphType>
    void check_global_task() {
        GraphType graph_data;
        const auto g = create_graph<GraphType>();
        std::int64_t global_triangle_count = graph_data.get_global_triangle_count();

        auto q = get_queue();

        std::allocator<char> alloc;
        const auto tc_desc = dal::preview::triangle_counting::descriptor<
            float,
            dal::preview::triangle_counting::method::ordered_count,
            dal::preview::triangle_counting::task::global,
            std::allocator<char>>(alloc);

        const auto result = dal::preview::vertex_ranking(q, tc_desc, g);
        REQUIRE(result.get_global_rank() == global_triangle_count);
    }
};

TEST_M(triangle_counting_gpu_test, "GPU: Local task") {
    this->check_local_task<complete_graph_5_type>();
    this->check_local_task<complete_graph_9_type>();
    this->check_local_task<triangle_graph_type>();
    this->check_local_task<two_vertices_graph_type>();
    this->check_local_task<wheel_graph_6_type>();
    this->check_local_task<graph_with_isolated_vertices_10_type>();
}

TEST_M(triangle_counting_gpu_test, "GPU: Local and global task") {
    this->check_local_and_global_task<complete_graph_5_type>();
    this->check_local_and_global_task<complete_graph_9_type>();
    this->check_local_and_global_task<triangle_graph_type>();
    this->check_local_and_global_task<two_vertices_graph_type>();
    this->check_local_and_global_task<wheel_graph_6_type>();
    this->check_local_and_global_task<graph_with_isolated_vertices_10_type>();
}

TEST_M(triangle_counting_gpu_test, "GPU: Global task") {
    this->check_global_task<complete_graph_5_type>();
    this->check_global_task<complete_graph_9_type>();
    this->check_global_task<triangle_graph_type>();
    this->check_global_task<two_vertices_graph_type>();
    this->check_global_task<wheel_graph_6_type>();
    this->check_global_task<graph_with_isolated_vertices_10_type>();
}

TEST_M(triangle_counting_gpu_test, "GPU: Null graph - local task") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    auto q = get_queue();

    std::allocator<char> alloc;
    const auto tc_desc = dal::preview::triangle_counting::descriptor<
        float,
        dal::preview::triangle_counting::method::ordered_count,
        dal::preview::triangle_counting::task::local,
        std::allocator<char>>(alloc);

    const auto result = dal::preview::vertex_ranking(q, tc_desc, null_graph);
    const auto& local_triangles =
        static_cast<const dal::homogen_table&>(result.get_ranks());
    REQUIRE(local_triangles.has_data() == false);
}

TEST_M(triangle_counting_gpu_test, "GPU: Null graph - global task") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    auto q = get_queue();

    std::allocator<char> alloc;
    const auto tc_desc = dal::preview::triangle_counting::descriptor<
        float,
        dal::preview::triangle_counting::method::ordered_count,
        dal::preview::triangle_counting::task::global,
        std::allocator<char>>(alloc);

    const auto result = dal::preview::vertex_ranking(q, tc_desc, null_graph);
    REQUIRE(result.get_global_rank() == 0);
}

TEST_M(triangle_counting_gpu_test, "GPU: Null graph - local_and_global task") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    auto q = get_queue();

    std::allocator<char> alloc;
    const auto tc_desc = dal::preview::triangle_counting::descriptor<
        float,
        dal::preview::triangle_counting::method::ordered_count,
        dal::preview::triangle_counting::task::local_and_global,
        std::allocator<char>>(alloc);

    const auto result = dal::preview::vertex_ranking(q, tc_desc, null_graph);
    const auto& local_triangles =
        static_cast<const dal::homogen_table&>(result.get_ranks());
    REQUIRE(local_triangles.has_data() == false);
    REQUIRE(result.get_global_rank() == 0);
}

TEST_M(triangle_counting_gpu_test, "GPU: topology_to_device round-trip") {
    const auto g = create_graph<complete_graph_5_type>();
    auto q = get_queue();

    // Access the internal topology
    const auto& topo = oneapi::dal::detail::get_impl(g).get_topology();

    // Transfer topology to device
    auto device_topo =
        dal::preview::detail::topology_to_device<std::int32_t>(q, topo);

    REQUIRE(device_topo.get_vertex_count() == 5);
    REQUIRE(device_topo.get_edge_count() == 10);
    REQUIRE(device_topo.get_rows() != nullptr);
    REQUIRE(device_topo.get_cols() != nullptr);

    // Transfer back to host and verify
    auto host_topo = dal::preview::detail::topology_to_host(device_topo);

    REQUIRE(host_topo.get_vertex_count() == 5);
    REQUIRE(host_topo.get_edge_count() == 10);

    // Verify row offsets match original
    for (std::int64_t i = 0; i <= 5; ++i) {
        REQUIRE(host_topo._rows[i] == topo._rows[i]);
    }

    // Verify column indices match original
    for (std::int64_t i = 0; i < 20; ++i) {
        REQUIRE(host_topo._cols[i] == topo._cols[i]);
    }
}

TEST_M(triangle_counting_gpu_test, "GPU: Empty topology round-trip") {
    auto q = get_queue();

    dal::preview::detail::topology<std::int32_t> empty_topo;

    auto device_topo =
        dal::preview::detail::topology_to_device<std::int32_t>(q, empty_topo);

    REQUIRE(device_topo.get_vertex_count() == 0);
    REQUIRE(device_topo.get_edge_count() == 0);

    auto host_topo = dal::preview::detail::topology_to_host(device_topo);

    REQUIRE(host_topo.get_vertex_count() == 0);
    REQUIRE(host_topo.get_edge_count() == 0);
}

} // namespace oneapi::dal::algo::triangle_counting::gpu::test
