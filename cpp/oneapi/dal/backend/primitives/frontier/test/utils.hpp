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

#pragma once

#include <random>
#include <vector>

void print_device_name(sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::cout << "Running on device: " << device_name << std::endl;
}

namespace oneapi::dal::preview::backend::primitives {
struct csr_graph_data {
    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint32_t> col_indices;
    std::vector<std::uint32_t> weights;
};

csr_graph_data generate_random_graph(std::uint64_t vertex_count,
                                     double edge_probability,
                                     std::uint32_t seed) {
    if (vertex_count == 0) {
        return { std::vector<std::uint32_t>{ 0 }, {}, {} };
    }

    std::mt19937 rng(seed);
    std::bernoulli_distribution edge_dist(edge_probability);
    std::uniform_int_distribution<std::uint32_t> vertex_dist(
        0,
        static_cast<std::uint32_t>(vertex_count - 1));

    std::vector<std::uint32_t> row_ptr(vertex_count + 1, 0);
    std::vector<std::uint32_t> col_indices;
    col_indices.reserve(vertex_count);

    for (std::uint64_t src = 0; src < vertex_count; ++src) {
        const auto edges_before = col_indices.size();
        for (std::uint64_t dst = 0; dst < vertex_count; ++dst) {
            if (src == dst) {
                continue;
            }
            if (edge_dist(rng)) {
                col_indices.push_back(static_cast<std::uint32_t>(dst));
            }
        }

        if (col_indices.size() == edges_before && vertex_count > 1) {
            std::uint32_t fallback = vertex_dist(rng);
            if (fallback == src) {
                fallback = (fallback + 1) % static_cast<std::uint32_t>(vertex_count);
            }
            col_indices.push_back(fallback);
        }

        row_ptr[src + 1] = static_cast<std::uint32_t>(col_indices.size());
    }

    std::vector<std::uint32_t> weights(col_indices.size(), 1);
    return { std::move(row_ptr), std::move(col_indices), std::move(weights) };
}

} // namespace oneapi::dal::preview::backend::primitives
