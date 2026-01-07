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

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/connected_components.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

int main(int argc, char** argv) {
    const auto filename = get_data_path("data/graph.csv");

    // read the graph
    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });

    // set algorithm parameters
    const auto cc_desc = dal::preview::connected_components::descriptor<>();

    // compute connected components
    const auto result_connected_components = dal::preview::vertex_partitioning(cc_desc, graph);

    // extract the result
    std::cout << "Components' labels:\n" << result_connected_components.get_labels() << std::endl;
    std::cout << "Number of connected components: "
              << result_connected_components.get_component_count() << std::endl;
    return 0;
}
