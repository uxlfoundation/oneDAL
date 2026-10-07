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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/linear_regression.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace result_options = dal::linear_regression::result_options;

// Trains a model on the GPU and runs inference with it on the host CPU.
// The trained model holds device-resident tables, which are copied to the
// host when the CPU inference converts them.
void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("data/linear_regression_train_data.csv");
    const auto train_response_file_name =
        get_data_path("data/linear_regression_train_responses.csv");
    const auto test_data_file_name = get_data_path("data/linear_regression_test_data.csv");
    const auto test_response_file_name = get_data_path("data/linear_regression_test_responses.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto lr_desc = dal::linear_regression::descriptor<>().set_result_options(
        result_options::coefficients | result_options::intercept);

    const auto train_result = dal::train(q, lr_desc, x_train, y_train);

    std::cout << "Coefficients:\n" << train_result.get_coefficients() << std::endl;
    std::cout << "Intercept:\n" << train_result.get_intercept() << std::endl;

    const auto lr_model = train_result.get_model();

    // No queue is passed, so inference runs on the host CPU
    const auto test_result = dal::infer(lr_desc, x_test, lr_model);

    std::cout << "Test results:\n" << test_result.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;
}

int main(int argc, char const* argv[]) {
    std::vector<sycl::device> devices;
    try_add_device(devices, &sycl::gpu_selector_v);
    for (auto d : devices) {
        std::cout << "Training on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << "Inferring on host CPU\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
