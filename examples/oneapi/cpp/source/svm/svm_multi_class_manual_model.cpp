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

#include <cmath>
#include <iostream>

#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    const auto train_data_file_name = get_data_path("data/svm_multi_class_train_dense_data.csv");
    const auto train_response_file_name =
        get_data_path("data/svm_multi_class_train_dense_label.csv");
    const auto test_data_file_name = get_data_path("data/svm_multi_class_test_dense_data.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });

    constexpr std::int64_t class_count = 5;
    constexpr std::int64_t expected_pairs = class_count * (class_count - 1) / 2;

    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
    const auto svm_desc =
        dal::svm::descriptor{ kernel_desc }.set_class_count(class_count).set_c(1.0);

    const auto trained_model = dal::train(svm_desc, x_train, y_train).get_model();

    if (trained_model.get_class_count() != class_count) {
        std::cerr << "FAIL: train should set class_count on the model" << std::endl;
        return 1;
    }
    if (trained_model.get_n_support_per_class().get_column_count() != class_count) {
        std::cerr << "FAIL: train should populate n_support_per_class" << std::endl;
        return 1;
    }

    const auto reference = dal::infer(svm_desc, trained_model, x_test).get_decision_function();
    if (reference.get_column_count() != expected_pairs) {
        std::cerr << "FAIL: multiclass decision_function must have " << expected_pairs << " columns"
                  << std::endl;
        return 1;
    }
    std::cout << "Trained-model decision_function shape: " << reference.get_row_count() << " x "
              << reference.get_column_count() << std::endl;

    // Reconstruct the model from public setters only - emulates the path
    // taken by move_estimator_to (no `train` call, no daal interop pointer).
    auto rebuilt = dal::svm::model<dal::svm::task::classification>{}
                       .set_class_count(class_count)
                       .set_support_vectors(trained_model.get_support_vectors())
                       .set_coeffs(trained_model.get_coeffs())
                       .set_biases(trained_model.get_biases())
                       .set_n_support_per_class(trained_model.get_n_support_per_class());

    const auto rebuilt_df = dal::infer(svm_desc, rebuilt, x_test).get_decision_function();
    std::cout << "Rebuilt-model decision_function shape: " << rebuilt_df.get_row_count() << " x "
              << rebuilt_df.get_column_count() << std::endl;
    if (rebuilt_df.get_column_count() != expected_pairs) {
        std::cerr << "FAIL: rebuilt model decision_function must have " << expected_pairs
                  << " columns" << std::endl;
        return 1;
    }

    // Numerical sanity: the reconstructed pair sub-models should give the same
    // decision values as the trained model.
    const auto ref_acc = dal::row_accessor<const float>{ reference }.pull();
    const auto reb_acc = dal::row_accessor<const float>{ rebuilt_df }.pull();
    if (ref_acc.get_count() != reb_acc.get_count()) {
        std::cerr << "FAIL: decision_function size mismatch" << std::endl;
        return 1;
    }
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (std::int64_t i = 0; i < ref_acc.get_count(); ++i) {
        const double d =
            std::abs(static_cast<double>(ref_acc[i]) - static_cast<double>(reb_acc[i]));
        sum_abs_diff += d;
        if (d > max_abs_diff)
            max_abs_diff = d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(ref_acc.get_count());
    std::cout << "max  |df_trained - df_rebuilt| = " << max_abs_diff << std::endl;
    std::cout << "mean |df_trained - df_rebuilt| = " << mean_abs_diff << std::endl;
    if (max_abs_diff > 1e-2) {
        std::cerr << "FAIL: rebuilt decision values differ from trained" << std::endl;
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
