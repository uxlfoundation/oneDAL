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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

/* Decision forest parameters */
const std::int64_t class_count = 5; /* Number of classes */

/** Visitor class, prints out tree nodes of the model when it is called back by model traversal method */
struct print_node_visitor {
    bool operator()(const df::leaf_node_info<df::task::classification>& info) {
        std::cout << std::string(info.get_level() * 2, ' ');
        std::cout << "Level " << info.get_level()
                  << ", leaf node. Response value = " << info.get_response()
                  << ", Impurity = " << info.get_impurity()
                  << ", Number of samples = " << info.get_sample_count() << ", Probabilities = { ";
        for (std::int64_t index_class = 0; index_class < class_count; ++index_class) {
            std::cout << info.get_probability(index_class) << ' ';
        }
        std::cout << "}" << std::endl;
        return true;
    }

    bool operator()(const df::split_node_info<df::task::classification>& info) {
        std::cout << std::string(info.get_level() * 2, ' ');
        std::cout << "Level " << info.get_level()
                  << ", split node. Feature index = " << info.get_feature_index()
                  << ", feature value = " << info.get_feature_value()
                  << ", Impurity = " << info.get_impurity()
                  << ", Number of samples = " << info.get_sample_count() << std::endl;
        return true;
    }
};

template <typename Task>
void print_model(const df::model<Task>& m) {
    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;
        m.traverse_depth_first(i, print_node_visitor{});
    }
}

void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("dev/data/df_classification_train_data.csv");
    const auto train_response_file_name =
        get_data_path("dev/data/df_classification_train_label.csv");
    const auto test_data_file_name = get_data_path("dev/data/df_classification_test_data.csv");
    const auto test_response_file_name = get_data_path("dev/data/df_classification_test_label.csv");

    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(q, dal::csv::data_source{ train_response_file_name });

    const auto x_test = dal::read<dal::table>(q, dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(q, dal::csv::data_source{ test_response_file_name });

    const auto df_desc = df::descriptor<float, df::method::hist, df::task::classification>{}
                             .set_class_count(class_count)
                             .set_tree_count(2)
                             .set_features_per_node(1)
                             .set_min_observations_in_leaf_node(8)
                             .set_min_observations_in_split_node(16)
                             .set_min_weight_fraction_in_leaf_node(0.0)
                             .set_min_impurity_decrease_in_split_node(0.0)
                             .set_max_tree_depth(15);

    try {
        const auto result_train = dal::train(q, df_desc, x_train, y_train);
        print_model(result_train.get_model());
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
