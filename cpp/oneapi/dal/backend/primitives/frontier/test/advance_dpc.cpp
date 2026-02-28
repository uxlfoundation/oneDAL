/*******************************************************************************
* Copyright contributors to the oneDAL project
* Copyright 2025 University of Salerno
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

#include "oneapi/dal/backend/primitives/frontier/frontier.hpp"
#include "oneapi/dal/backend/primitives/frontier/advance.hpp"
#include "oneapi/dal/backend/primitives/frontier/graph.hpp"
#include "oneapi/dal/backend/primitives/frontier/test/utils.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::preview::backend::primitives::test {

namespace pr = dal::backend::primitives;

template <typename T>
void print_frontier(const T* data, std::uint64_t count, std::uint64_t num_items) {
    std::uint64_t element_bitsize = sizeof(T) * 8;
    for (std::uint64_t i = 0; i < count; ++i) {
        for (std::uint64_t j = 0; j < element_bitsize; ++j) {
            std::cout << ((data[i] & (static_cast<T>(1) << j)) ? "1" : "0");
        }
    }
}

std::vector<bool> compute_next_frontier(const std::vector<std::uint32_t>& row_ptr,
                                        const std::vector<std::uint32_t>& col_indices,
                                        const std::vector<bool>& frontier) {
    std::vector<bool> next_frontier(frontier.size(), false);

    for (std::uint64_t node = 0; node < frontier.size(); ++node) {
        if (!frontier[node])
            continue;

        auto start = row_ptr[node];
        auto end = row_ptr[node + 1];

        for (std::uint64_t i = start; i < end; ++i) {
            auto neighbor = col_indices[i];
            next_frontier[neighbor] = true;
        }
    }
    return next_frontier;
}

template <typename T>
void compare_frontiers(T& device_frontier,
                       const std::vector<bool>& host_frontier,
                       std::uint64_t num_nodes) {
    for (std::uint64_t i = 0; i < num_nodes; ++i) {
        bool tmpd = device_frontier.check(i);
        bool tmph = host_frontier[i];
        if (tmpd != tmph) {
            std::cout << "Mismatch at vertex " << i << ": device = " << tmpd << ", host = " << tmph
                      << std::endl;
        }
        REQUIRE(tmpd == tmph);
    }
}

TEST("test advance operation", "[advance]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    const std::uint32_t seed = GENERATE(42u, 313u, 2025u);
    const double edge_probability = GENERATE(0.05, 0.1, 0.2);
    const std::uint64_t num_nodes = GENERATE(128, 512, 1024);

    const auto graph_data = generate_random_graph(num_nodes, edge_probability, seed);
    auto graph = csr_graph(queue, graph_data.row_ptr, graph_data.col_indices, graph_data.weights);
    auto in_frontier =
        frontier<std::uint32_t>(queue, graph.get_vertex_count(), sycl::usm::alloc::device);
    auto out_frontier =
        frontier<std::uint32_t>(queue, graph.get_vertex_count(), sycl::usm::alloc::device);

    std::vector<bool> host_frontier(num_nodes, false);
    std::mt19937 frontier_rng(seed ^ 0x9e3779b9u);
    std::bernoulli_distribution frontier_dist(0.1);

    for (std::uint64_t vertex = 0; vertex < num_nodes; ++vertex) {
        if (frontier_dist(frontier_rng)) {
            in_frontier.insert(static_cast<std::uint32_t>(vertex));
            host_frontier[vertex] = true;
        }
    }

    if (in_frontier.empty()) {
        std::uniform_int_distribution<std::uint32_t> vertex_dist(
            0,
            static_cast<std::uint32_t>(num_nodes - 1));
        const auto fallback_vertex = vertex_dist(frontier_rng);
        in_frontier.insert(fallback_vertex);
        host_frontier[fallback_vertex] = true;
    }

    advance(graph,
            in_frontier,
            out_frontier,
            [=](auto vertex, auto neighbor, auto edge, auto weight) {
                return true; // Always advance
            })
        .wait_and_throw();

    auto tmp_frontier =
        compute_next_frontier(graph_data.row_ptr, graph_data.col_indices, host_frontier);
    compare_frontiers(out_frontier, tmp_frontier, num_nodes);
} // TEST "test advance operation"

} // namespace oneapi::dal::preview::backend::primitives::test
