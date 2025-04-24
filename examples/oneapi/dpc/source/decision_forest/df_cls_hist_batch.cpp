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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("df_classification_train_data.csv");
    const auto train_response_file_name = get_data_path("df_classification_train_label.csv");
    const auto test_data_file_name = get_data_path("df_classification_test_data.csv");
    const auto test_response_file_name = get_data_path("df_classification_test_label.csv");

    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(q, dal::csv::data_source{ train_response_file_name });

    const auto x_test = dal::read<dal::table>(q, dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(q, dal::csv::data_source{ test_response_file_name });

    const auto df_desc =
        df::descriptor<float, df::method::hist, df::task::classification>{}
            .set_class_count(5)
            .set_tree_count(10)
            .set_features_per_node(x_train.get_column_count())
            .set_min_observations_in_leaf_node(8)
            .set_min_observations_in_split_node(16)
            .set_min_weight_fraction_in_leaf_node(0.0)
            .set_min_impurity_decrease_in_split_node(0.0)
            .set_error_metric_mode(df::error_metric_mode::out_of_bag_error)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_infer_mode(df::infer_mode::class_responses | df::infer_mode::class_probabilities)
            .set_voting_mode(df::voting_mode::weighted);

    try {
        const auto result_train = dal::train(q, df_desc, x_train, y_train);

        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;

        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;

        const auto result_infer = dal::infer(q, df_desc, result_train.get_model(), x_test);

        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n" << result_infer.get_probabilities() << std::endl;

        std::cout << "Ground truth:\n" << y_test << std::endl;
    }
    catch (dal::unimplemented& e) {
        std::cout << "  " << e.what() << std::endl;
        return;
    }
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
