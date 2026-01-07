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

#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/common.hpp"

namespace dal = oneapi::dal;

int main(int argc, char **argv) {
    auto target_filename = get_data_path("dev/data/si_target_graph.csv");
    auto pattern_filename = get_data_path("dev/data/si_pattern_graph.csv");

    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;

    const auto target_graph = dal::read<graph_t>(dal::csv::data_source{ target_filename });
    const auto pattern_graph = dal::read<graph_t>(dal::csv::data_source{ pattern_filename });

    // set algorithm parameters
    const auto subgraph_isomorphism_desc =
        dal::preview::subgraph_isomorphism::descriptor<>()
            .set_kind(dal::preview::subgraph_isomorphism::kind::non_induced)
            .set_semantic_match(false)
            .set_max_match_count(10);

    const auto result =
        dal::preview::graph_matching(subgraph_isomorphism_desc, target_graph, pattern_graph);

    // extract the result
    std::cout << "Number of matchings: " << result.get_match_count() << std::endl;
    std::cout << "Matchings:" << std::endl << result.get_vertex_match() << std::endl;

    return 0;
}
