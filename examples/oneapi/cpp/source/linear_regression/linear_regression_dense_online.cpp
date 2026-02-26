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

#include "oneapi/dal/algo/linear_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace result_options = dal::linear_regression::result_options;

void run() {
    const auto train_data_file_name = get_data_path("data/linear_regression_train_data.csv");
    const auto train_response_file_name =
        get_data_path("data/linear_regression_train_responses.csv");
    const auto test_data_file_name = get_data_path("data/linear_regression_test_data.csv");
    const auto test_response_file_name = get_data_path("data/linear_regression_test_responses.csv");
    const std::int64_t nBlocks = 10;
    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto lr_desc = dal::linear_regression::descriptor<>().set_result_options(
        result_options::coefficients | result_options::intercept);

    dal::linear_regression::partial_train_result<> partial_result;

    auto input_table_x = split_table_by_rows<double>(x_train, nBlocks);
    auto input_table_y = split_table_by_rows<double>(y_train, nBlocks);
    for (std::int64_t i = 0; i < nBlocks; i++) {
        partial_result =
            dal::partial_train(lr_desc, partial_result, input_table_x[i], input_table_y[i]);
    }
    auto result = dal::finalize_train(lr_desc, partial_result);

    std::cout << "Coefficients:\n" << result.get_coefficients() << std::endl;
    std::cout << "Intercept:\n" << result.get_intercept() << std::endl;

    const auto lr_model = result.get_model();

    const auto test_result = dal::infer(lr_desc, x_test, lr_model);

    std::cout << "Test results:\n" << test_result.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;
}

int main(int argc, char const* argv[]) {
    run();
    return 0;
}
