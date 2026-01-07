/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue &q) {
    const auto train_data_file_name = get_data_path("data/kmeans_csr_train_data.csv");
    const auto initial_centroids_file_name =
        get_data_path("data/kmeans_dense_train_centroids.csv");
    const auto test_data_file_name = get_data_path("data/kmeans_csr_test_data.csv");
    const auto test_response_file_name = get_data_path("data/kmeans_dense_test_label.csv");

    const auto x_train =
        dal::read<dal::csr_table>(q,
                                  dal::csv::data_source<float>{ train_data_file_name },
                                  dal::csv::read_args<dal::csr_table>()
                                      .set_sparse_indexing(dal::sparse_indexing::one_based)
                                      .set_feature_count(20));
    const auto initial_centroids =
        dal::read<dal::table>(q, dal::csv::data_source{ initial_centroids_file_name });

    const auto x_test =
        dal::read<dal::csr_table>(q,
                                  dal::csv::data_source<float>{ test_data_file_name },
                                  dal::csv::read_args<dal::csr_table>()
                                      .set_sparse_indexing(dal::sparse_indexing::one_based)
                                      .set_feature_count(20));
    const auto y_test = dal::read<dal::table>(q, dal::csv::data_source{ test_response_file_name });

    const auto kmeans_desc = dal::kmeans::descriptor<float, dal::kmeans::method::lloyd_csr>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = dal::train(q, kmeans_desc, x_train, initial_centroids);

    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "Responses:\n" << result_train.get_responses() << std::endl;
    std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;

    const auto result_test = dal::infer(q, kmeans_desc, result_train.get_model(), x_test);

    std::cout << "Infer result:\n" << result_test.get_responses() << std::endl;

    std::cout << "Ground truth:\n" << y_test << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
