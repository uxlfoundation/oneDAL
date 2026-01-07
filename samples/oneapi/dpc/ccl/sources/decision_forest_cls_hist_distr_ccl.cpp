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

#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

void run(sycl::queue &queue) {
    const auto train_data_file_name = get_data_path("data/df_classification_train_data.csv");
    const auto train_response_file_name =
        get_data_path("data/df_classification_train_label.csv");
    const auto test_data_file_name = get_data_path("data/df_classification_test_data.csv");
    const auto test_response_file_name = get_data_path("data/df_classification_test_label.csv");

    const auto x_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(queue, dal::csv::data_source{ test_data_file_name });
    const auto y_test =
        dal::read<dal::table>(queue, dal::csv::data_source{ test_response_file_name });

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto x_train_vec = split_table_by_rows<float>(queue, x_train, rank_count);
    auto y_train_vec = split_table_by_rows<float>(queue, y_train, rank_count);
    auto x_test_vec = split_table_by_rows<float>(queue, x_test, rank_count);
    auto y_test_vec = split_table_by_rows<float>(queue, y_test, rank_count);

    const auto df_desc =
        df::descriptor<float, df::method::hist, df::task::classification>{}
            .set_class_count(5)
            .set_tree_count(50)
            .set_features_per_node(x_train.get_column_count())
            .set_min_observations_in_leaf_node(8)
            .set_min_observations_in_split_node(16)
            .set_min_weight_fraction_in_leaf_node(0.0)
            .set_min_impurity_decrease_in_split_node(0.0)
            .set_error_metric_mode(df::error_metric_mode::out_of_bag_error)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_infer_mode(df::infer_mode::class_responses | df::infer_mode::class_probabilities)
            .set_voting_mode(df::voting_mode::weighted);

    const auto result_train =
        dal::preview::train(comm, df_desc, x_train_vec[rank_id], y_train_vec[rank_id]);

    if (comm.get_rank() == 0) {
        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;

        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
    }

    const auto result_infer =
        dal::preview::infer(comm, df_desc, result_train.get_model(), x_test_vec[rank_id]);

    if (comm.get_rank() == 0) {
        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n" << result_infer.get_probabilities() << std::endl;

        std::cout << "Ground truth:\n" << y_test_vec[rank_id] << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    ccl::init();
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    auto device = sycl::device(sycl::gpu_selector_v);
    std::cout << "Running on " << device.get_platform().get_info<sycl::info::platform::name>()
              << ", " << device.get_info<sycl::info::device::name>() << std::endl;
    sycl::queue q{ device };
    run(q);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
