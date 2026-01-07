/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace svm = dal::svm;

int main(int argc, char const *argv[]) {
    const auto train_data_file_name = get_data_path("data/svm_two_class_train_sparse_data.csv");
    const auto train_response_file_name =
        get_data_path("data/svm_two_class_train_sparse_labels.csv");
    const auto test_data_file_name = get_data_path("data/svm_two_class_test_sparse_data.csv");
    const auto test_response_file_name = get_data_path("data/svm_two_class_test_sparse_labels.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });

    // Convert data table to CSR table
    const auto x_train_csr = convert_to_csr<float>(x_train);

    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);

    const auto svm_desc =
        svm::descriptor<float, svm::method::thunder, svm::task::nu_classification>{ kernel_desc }
            .set_nu(0.5)
            .set_accuracy_threshold(0.001)
            .set_max_iteration_count(100)
            .set_cache_size(200.0)
            .set_tau(1e-6);

    const auto result_train = dal::train(svm_desc, x_train, y_train);

    std::cout << "Biases:\n" << result_train.get_biases() << std::endl;
    std::cout << "Support indices:\n" << result_train.get_support_indices() << std::endl;

    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_true = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto result_infer = dal::infer(svm_desc, result_train.get_model(), x_test);

    std::cout << "Decision function result:\n" << result_infer.get_decision_function() << std::endl;
    std::cout << "Responses result:\n" << result_infer.get_responses() << std::endl;
    std::cout << "Responses true:\n" << y_true << std::endl;

    return 0;
}
