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
#include <cmath>
#include <limits>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <sycl/sycl.hpp>

#include "oneapi/dal/algo/shortest_paths/traverse.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::shortest_paths::gpu::test {

namespace dal = oneapi::dal;

constexpr double unreachable_double = std::numeric_limits<double>::max();
constexpr std::int32_t unreachable_int32 = std::numeric_limits<std::int32_t>::max();

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
    std::int64_t get_source() const {
        return source;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
    std::int64_t source;
};

class d_isolated_graph_type : public graph_base_data {
public:
    d_isolated_graph_type() {
        vertex_count = 10;
        edge_count = 0;
        cols_count = 0;
        rows_count = 11;
        source = 0;
    }
    std::array<std::int64_t, 11> rows = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<double, 0> edge_weights = {};
    std::array<double, 10> distances = { 0,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double,
                                         unreachable_double };
};

class d_isolated_int_graph_type : public graph_base_data {
public:
    d_isolated_int_graph_type() {
        vertex_count = 10;
        edge_count = 0;
        cols_count = 0;
        rows_count = 11;
        source = 0;
    }
    std::array<std::int64_t, 11> rows = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<std::int32_t, 0> edge_weights = {};
    std::array<std::int32_t, 10> distances = { 0,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32,
                                               unreachable_int32 };
};

class d_one_bucket_graph_type : public graph_base_data {
public:
    d_one_bucket_graph_type() {
        vertex_count = 6;
        edge_count = 9;
        cols_count = 9;
        rows_count = 7;
        source = 0;
    }
    std::array<std::int64_t, 7> rows = { 0, 5, 6, 7, 8, 9, 9 };
    std::array<std::int32_t, 9> cols = { 1, 2, 3, 4, 5, 2, 3, 4, 5 };
    std::array<double, 9> edge_weights = { 1, 3, 4, 5, 6, 1, 1, 1, 1 };
    std::array<double, 6> distances = { 0, 1, 2, 3, 4, 5 };
};

class d_graph_3_type : public graph_base_data {
public:
    d_graph_3_type() {
        vertex_count = 21;
        edge_count = 49;
        cols_count = 49;
        rows_count = 22;
        source = 0;
    }
    std::array<std::int64_t, 22> rows = { 0,  3,  6,  8,  11, 14, 15, 18, 21, 22, 25,
                                          27, 30, 32, 35, 37, 39, 42, 42, 45, 47, 49 };
    std::array<std::int32_t, 49> cols = { 1,  2,  3,  2,  5,  13, 5,  6,  2,  4,  7,  0,  7,
                                          8,  6,  3,  7,  9,  6,  10, 11, 12, 5,  13, 14, 6,
                                          9,  10, 14, 16, 7,  11, 14, 17, 18, 10, 15, 11, 19,
                                          12, 15, 20, 14, 17, 19, 14, 20, 8,  15 };
    std::array<double, 49> edge_weights = { 95, 89, 70, 19, 96, 73, 42, 19, 23, 40, 10, 70, 18,
                                            64, 47, 94, 21, 22, 26, 40, 66, 91, 62, 80, 10, 57,
                                            63, 99, 73, 17, 12, 14, 42, 92, 69, 39, 58, 38, 10,
                                            87, 36, 71, 45, 93, 11, 21, 21, 66, 46 };
    std::array<double, 21> distances = { 0,   95,  89,  70,  110, 131, 106, 80,  174, 128, 120,
                                         146, 250, 168, 138, 196, 163, 260, 237, 206, 227 };
};

template <typename EdgeValue, typename GraphT>
void check_distances(sycl::queue& queue,
                     const GraphT& graph,
                     std::int64_t source,
                     const EdgeValue* expected_distances,
                     std::int64_t vertex_count,
                     double delta) {
    using namespace dal::preview::shortest_paths;
    const auto desc =
        descriptor<float, method::delta_stepping, task::one_to_all>(source,
                                                                    delta,
                                                                    optional_results::distances);

    const auto result = dal::preview::traverse(queue, desc, graph);
    const auto& dist_table = result.get_distances();

    REQUIRE(dist_table.get_row_count() == vertex_count);
    REQUIRE(dist_table.get_column_count() == 1);

    auto dist_accessor = dal::row_accessor<const EdgeValue>(dist_table).pull();
    const EdgeValue* dist_data = dist_accessor.get_data();

    for (std::int64_t i = 0; i < vertex_count; ++i) {
        if (expected_distances[i] == std::numeric_limits<EdgeValue>::max()) {
            REQUIRE(dist_data[i] == std::numeric_limits<EdgeValue>::max());
        }
        else {
            REQUIRE(std::abs(static_cast<double>(dist_data[i]) -
                             static_cast<double>(expected_distances[i])) < 1e-6);
        }
    }
}

TEST_CASE("shortest_paths GPU: isolated vertices (double)", "[shortest_paths][gpu]") {
    DECLARE_TEST_POLICY(policy);
    auto selector = sycl::gpu_selector_v;
    auto queue = sycl::queue(selector);

    if (!queue.get_device().has(sycl::aspect::fp64)) {
        WARN("Device does not support fp64, skipping");
        return;
    }

    d_isolated_graph_type data;
    const auto builder =
        dal::preview::detail::directed_adjacency_vector_graph_builder<std::int32_t,
                                                                      double,
                                                                      dal::preview::empty_value,
                                                                      int>(
            data.get_vertex_count(),
            data.get_edge_count(),
            data.rows.data(),
            data.cols.data(),
            data.edge_weights.data());
    const auto& graph = builder.get_graph();

    check_distances<double>(queue,
                            graph,
                            data.get_source(),
                            data.distances.data(),
                            data.get_vertex_count(),
                            1.0);
}

TEST_CASE("shortest_paths GPU: isolated vertices (int32)", "[shortest_paths][gpu]") {
    DECLARE_TEST_POLICY(policy);
    auto selector = sycl::gpu_selector_v;
    auto queue = sycl::queue(selector);

    d_isolated_int_graph_type data;
    const auto builder =
        dal::preview::detail::directed_adjacency_vector_graph_builder<std::int32_t,
                                                                      std::int32_t,
                                                                      dal::preview::empty_value,
                                                                      int>(
            data.get_vertex_count(),
            data.get_edge_count(),
            data.rows.data(),
            data.cols.data(),
            data.edge_weights.data());
    const auto& graph = builder.get_graph();

    check_distances<std::int32_t>(queue,
                                  graph,
                                  data.get_source(),
                                  data.distances.data(),
                                  data.get_vertex_count(),
                                  1.0);
}

TEST_CASE("shortest_paths GPU: one-bucket graph (double)", "[shortest_paths][gpu]") {
    DECLARE_TEST_POLICY(policy);
    auto selector = sycl::gpu_selector_v;
    auto queue = sycl::queue(selector);

    if (!queue.get_device().has(sycl::aspect::fp64)) {
        WARN("Device does not support fp64, skipping");
        return;
    }

    d_one_bucket_graph_type data;
    const auto builder =
        dal::preview::detail::directed_adjacency_vector_graph_builder<std::int32_t,
                                                                      double,
                                                                      dal::preview::empty_value,
                                                                      int>(
            data.get_vertex_count(),
            data.get_edge_count(),
            data.rows.data(),
            data.cols.data(),
            data.edge_weights.data());
    const auto& graph = builder.get_graph();

    check_distances<double>(queue,
                            graph,
                            data.get_source(),
                            data.distances.data(),
                            data.get_vertex_count(),
                            1.0);
}

TEST_CASE("shortest_paths GPU: graph_3 (21 vertices, double)", "[shortest_paths][gpu]") {
    DECLARE_TEST_POLICY(policy);
    auto selector = sycl::gpu_selector_v;
    auto queue = sycl::queue(selector);

    if (!queue.get_device().has(sycl::aspect::fp64)) {
        WARN("Device does not support fp64, skipping");
        return;
    }

    d_graph_3_type data;
    const auto builder =
        dal::preview::detail::directed_adjacency_vector_graph_builder<std::int32_t,
                                                                      double,
                                                                      dal::preview::empty_value,
                                                                      int>(
            data.get_vertex_count(),
            data.get_edge_count(),
            data.rows.data(),
            data.cols.data(),
            data.edge_weights.data());
    const auto& graph = builder.get_graph();

    check_distances<double>(queue,
                            graph,
                            data.get_source(),
                            data.distances.data(),
                            data.get_vertex_count(),
                            1.0);
}

} // namespace oneapi::dal::algo::shortest_paths::gpu::test
