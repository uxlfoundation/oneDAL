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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/logistic_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"
#include <chrono>

namespace dal = oneapi::dal;
namespace result_options = dal::logistic_regression::result_options;

auto now = std::chrono::steady_clock::now();

float get_time_duration(std::chrono::time_point<std::chrono::steady_clock>& a,
                        std::chrono::time_point<std::chrono::steady_clock>& b) {
    return (float)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() / 1000;
}

void run(sycl::queue& q) {
    const auto x_train_filename = get_data_path("dev/data/df_binary_classification_train_data.csv");
    const auto y_train_filename =
        get_data_path("dev/data/df_binary_classification_train_label.csv");
    const auto x_test_filename = get_data_path("dev/data/df_binary_classification_test_data.csv");
    const auto y_test_filename = get_data_path("dev/data/df_binary_classification_test_label.csv");

    auto tm1 = std::chrono::steady_clock::now();

    std::cout << "Loading dataset... ";

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ x_train_filename });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ y_train_filename });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ x_test_filename });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ y_test_filename });

    auto tm2 = std::chrono::steady_clock::now();
    std::cout << get_time_duration(tm1, tm2) << " s" << std::endl;

    std::cout << "Fitting model... ";

    using method_t = dal::logistic_regression::method::dense_batch;
    using task_t = dal::logistic_regression::task::classification;
    using optimizer_t = dal::newton_cg::descriptor<>;

    const auto optimizer_desc = dal::newton_cg::descriptor<>(1e-4, 10l);

    const auto log_reg_desc =
        dal::logistic_regression::descriptor<float, method_t, task_t, optimizer_t>(true,
                                                                                   0.5,
                                                                                   optimizer_desc)
            .set_result_options(result_options::coefficients | result_options::intercept |
                                result_options::iterations_count);

    const auto train_result = dal::train(q, log_reg_desc, x_train, y_train);

    auto tm3 = std::chrono::steady_clock::now();
    std::cout << get_time_duration(tm2, tm3) << " s" << std::endl;

    std::cout << "Coefficients:\n" << train_result.get_coefficients() << std::endl;
    std::cout << "Intercept:\n" << train_result.get_intercept() << std::endl;
    std::cout << "Iterations count: " << train_result.get_iterations_count() << std::endl;

    const auto log_reg_model = train_result.get_model();

    std::cout << "Inference... ";

    const auto test_result = dal::infer(q, log_reg_desc, x_test, log_reg_model);

    auto tm4 = std::chrono::steady_clock::now();
    std::cout << get_time_duration(tm3, tm4) << " s" << std::endl;

    std::cout << "Test results:\n" << test_result.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;

    auto y_true_arr = oneapi::dal::row_accessor<const std::int32_t>(y_test).pull();
    const auto gth_ptr = y_true_arr.get_data();

    auto pred_arr =
        oneapi::dal::row_accessor<const std::int32_t>(test_result.get_responses()).pull();
    const auto pred_ptr = pred_arr.get_data();

    std::int64_t acc = 0;

    for (std::int64_t i = 0; i < y_test.get_row_count(); ++i) {
        if (pred_ptr[i] == gth_ptr[i]) {
            acc += 1;
        }
    }

    std::cout << "Accuracy on test: " << double(acc) / y_test.get_row_count() << " (" << acc
              << " out of " << y_test.get_row_count() << ")" << std::endl;
}

int main(int argc, char const* argv[]) {
    std::vector<sycl::device> devices;
    try_add_device(devices, &sycl::gpu_selector_v);
    for (auto d : devices) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
