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

#include "oneapi/dal/algo/objective_function.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"
#include <iostream>

namespace dal = oneapi::dal;
namespace objective_function = dal::objective_function;
namespace logloss_objective = dal::logloss_objective;
namespace result_options = objective_function::result_options;

int main(int argc, char const *argv[]) {
    const auto data_filename = get_data_path("data/df_binary_classification_train_data.csv");
    const auto labels_filename = get_data_path("data/df_binary_classification_train_label.csv");
    const auto params_filename = get_data_path("data/logreg_params.csv");
    const auto data = dal::read<dal::table>(dal::csv::data_source{ data_filename });
    const auto labels = dal::read<dal::table>(dal::csv::data_source{ labels_filename });
    const auto params = dal::read<dal::table>(dal::csv::data_source{ params_filename });

    const double L1 = 0.0;
    const double L2 = 2.0;
    using dense_batch_method = objective_function::method::dense_batch;

    auto logloss_desc = logloss_objective::descriptor<float>(L1, L2).set_intercept_flag(true);

    auto desc = objective_function::descriptor<float, dense_batch_method>(logloss_desc)
                    .set_result_options(result_options::value | result_options::gradient |
                                        result_options::hessian);

    auto result = dal::compute(desc, data, params, labels);

    std::cout << "Logistic loss\n" << result.get_value() << std::endl;

    std::cout << "Gradient\n" << result.get_gradient() << std::endl;

    std::cout << "Hessian\n" << result.get_hessian() << std::endl;
}
