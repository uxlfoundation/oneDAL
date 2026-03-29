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

#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace knn = dal::knn;

int main(int argc, char const *argv[]) {
    const auto train_data_file_name = get_data_path("data/k_nearest_neighbors_train_data.csv");
    const auto query_data_file_name = get_data_path("data/k_nearest_neighbors_test_data.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto x_query = dal::read<dal::table>(dal::csv::data_source{ query_data_file_name });

    using cosine_desc_t = dal::cosine_distance::descriptor<float>;
    const auto cosine_desc = cosine_desc_t{};

    const std::size_t neighbors_count = 6;
    const auto knn_desc =
        knn::descriptor<float, knn::method::brute_force, knn::task::search, cosine_desc_t>(
            neighbors_count,
            cosine_desc);

    const auto train_result = dal::train(knn_desc, x_train);
    const auto test_result = dal::infer(knn_desc, x_query, train_result.get_model());

    std::cout << "Indices result:\n" << test_result.get_indices() << std::endl;
    std::cout << "Distance result:\n" << test_result.get_distances() << std::endl;
    return 0;
}
