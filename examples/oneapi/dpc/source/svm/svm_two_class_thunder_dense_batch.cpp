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

#include <sycl/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue &q) {
    const auto train_data_file_name = get_data_path("data/svm_two_class_train_dense_data.csv");
    const auto train_response_file_name =
        get_data_path("data/svm_two_class_train_dense_label.csv");
    const auto test_data_file_name = get_data_path("data/svm_two_class_test_dense_data.csv");
    const auto test_response_file_name =
        get_data_path("data/svm_two_class_test_dense_label.csv");

    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(q, dal::csv::data_source{ train_response_file_name });

    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
    const auto svm_desc = dal::svm::descriptor{ kernel_desc }
                              .set_c(1.0)
                              .set_accuracy_threshold(0.001)
                              .set_max_iteration_count(100)
                              .set_cache_size(200.0)
                              .set_tau(1e-6);

    const auto result_train = dal::train(q, svm_desc, x_train, y_train);

    std::cout << "Biases:\n" << result_train.get_biases() << std::endl;
    std::cout << "Support indices:\n" << result_train.get_support_indices() << std::endl;

    const auto x_test = dal::read<dal::table>(q, dal::csv::data_source{ test_data_file_name });
    const auto y_true = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto result_test = dal::infer(q, svm_desc, result_train.get_model(), x_test);

    std::cout << "Decision function result:\n" << result_test.get_decision_function() << std::endl;
    std::cout << "Responses result:\n" << result_test.get_responses() << std::endl;
    std::cout << "Responses true:\n" << y_true << std::endl;
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
