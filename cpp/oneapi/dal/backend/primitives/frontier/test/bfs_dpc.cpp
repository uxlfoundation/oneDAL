/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "oneapi/dal/backend/primitives/frontier.hpp"
#include "oneapi/dal/backend/primitives/frontier/test/utils.hpp"
#include "oneapi/dal/test/engine/common.hpp"

#include <queue>
#include <vector>
#include <map>

namespace oneapi::dal::backend::primitives::test {

namespace pr = dal::backend::primitives;

template <typename T>
std::vector<std::uint32_t> host_bfs(const std::vector<T>& row_offsets,
                                    const std::vector<T>& col_indices,
                                    const T src) {
    std::vector<std::uint32_t> distances(row_offsets.size() - 1,
                                         std::numeric_limits<std::uint32_t>::max());
    std::queue<T> q;
    q.push(src);
    distances[src] = 0;

    while (!q.empty()) {
        T node = q.front();
        q.pop();

        for (size_t i = row_offsets[node]; i < row_offsets[node + 1]; ++i) {
            auto neighbor = col_indices[i];
            if (distances[neighbor] == std::numeric_limits<std::uint32_t>::max()) {
                distances[neighbor] = distances[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return distances;
}

TEST("test BFS", "[bfs]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    const std::uint32_t seed = GENERATE(42u, 313u, 2025u);
    const double edge_probability = GENERATE(0.05, 0.1, 0.2);
    const std::size_t num_nodes = GENERATE(128, 512, 1024);

    const auto graph_data = generate_random_graph(num_nodes, edge_probability, seed);
    auto graph =
        pr::csr_graph(queue, graph_data.row_ptr, graph_data.col_indices, graph_data.weights);
    auto in_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    auto out_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    pr::ndarray<std::uint32_t, 1> distance =
        pr::ndarray<std::uint32_t, 1>::empty(queue,
                                             { static_cast<std::int64_t>(num_nodes) },
                                             sycl::usm::alloc::device);
    auto distance_ptr = distance.get_mutable_data();

    const std::uint32_t src = 0;

    queue
        .submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_nodes), [=](sycl::id<1> idx) {
                distance_ptr[idx] = idx[0] == src ? 0 : num_nodes + 1;
            });
        })
        .wait_and_throw();

    in_frontier.insert(src);
    size_t iter = 0;

    /// Start BFS
    while (!in_frontier.empty()) {
        auto e = pr::advance(graph,
                             in_frontier,
                             out_frontier,
                             [=](auto vertex, auto neighbor, auto edge, auto weight) {
                                 bool visited = distance_ptr[neighbor] < num_nodes + 1;
                                 if (!visited) {
                                     distance_ptr[neighbor] = iter + 1;
                                 }
                                 return !visited;
                             });
        e.wait_and_throw();
        iter++;
        pr::swap_frontiers(in_frontier, out_frontier);
        out_frontier.clear();
    }
    /// End BFS

    auto expected_distances =
        host_bfs<std::uint32_t>(graph_data.row_ptr, graph_data.col_indices, src);
    auto actual_distances = distance.to_host(queue).get_data();
    for (size_t i = 0; i < num_nodes; ++i) {
        REQUIRE(actual_distances[i] == expected_distances[i]);
    }
} // TEST "test advance operation"

} // namespace oneapi::dal::backend::primitives::test
