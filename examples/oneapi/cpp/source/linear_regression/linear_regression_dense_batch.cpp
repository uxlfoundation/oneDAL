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

#include "oneapi/dal/algo/linear_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace result_options = dal::linear_regression::result_options;

void run() {
    const auto train_data_file_name = get_data_path("dev/data/linear_regression_train_data.csv");
    const auto train_response_file_name =
        get_data_path("dev/data/linear_regression_train_responses.csv");
    const auto test_data_file_name = get_data_path("dev/data/linear_regression_test_data.csv");
    const auto test_response_file_name =
        get_data_path("dev/data/linear_regression_test_responses.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto lr_desc = dal::linear_regression::descriptor<>().set_result_options(
        result_options::coefficients | result_options::intercept);

    const auto train_result = dal::train(lr_desc, x_train, y_train);

    std::cout << "Coefficients:\n" << train_result.get_coefficients() << std::endl;
    std::cout << "Intercept:\n" << train_result.get_intercept() << std::endl;

    const auto lr_model = train_result.get_model();

    const auto test_result = dal::infer(lr_desc, x_test, lr_model);

    std::cout << "Test results:\n" << test_result.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;
}

int main(int argc, char const* argv[]) {
    run();
    return 0;
}
