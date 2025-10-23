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

#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/classification.hpp"

#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::svm::test {

namespace te = dal::test::engine;
namespace rbf = dal::rbf_kernel;
namespace linear = dal::linear_kernel;
namespace polynomial = dal::polynomial_kernel;
namespace sigmoid = dal::sigmoid_kernel;

template <typename TestType>
class svm_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using KernelTypeLinear = linear::descriptor<Float, linear::method::dense>;
    using KernelTypeRBF = rbf::descriptor<Float, rbf::method::dense>;
    using KernelTypePolynomial = polynomial::descriptor<Float, polynomial::method::dense>;
    using KernelTypeSigmoid = sigmoid::descriptor<Float, sigmoid::method::dense>;

    bool not_available_on_device() {
        constexpr bool is_smo = std::is_same_v<Method, svm::method::smo>;
        return this->get_policy().is_gpu() && is_smo;
    }

    bool kernel_not_available_on_device() {
        return this->get_policy().is_gpu();
    }

    bool weights_not_available_on_device() {
        return this->get_policy().is_gpu();
    }

    bool multiclass_not_available_on_device() {
        return this->get_policy().is_gpu();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    template <class KernelType>
    void check_kernel(
        const table& train_data,
        const table& train_responses,
        const svm::descriptor<Float, Method, svm::task::classification, KernelType>& desc,
        const std::int64_t support_vector_count,
        const table& support_indices,
        const table& decision_function,
        const table& responses) {
        CAPTURE(support_vector_count);

        INFO("run training");
        auto train_result = this->train(desc, train_data, train_responses);
        const auto model = train_result.get_model();
        check_train_result(train_data, train_result, support_vector_count, support_indices);

        INFO("run inference");
        const auto infer_result = this->infer(desc, model, train_data);
        check_infer_result(train_data, infer_result, decision_function, responses);
    }

    void check_different_responses(
        const table& train_data,
        const table& train_responses,
        const svm::descriptor<Float, Method, svm::task::classification, KernelTypeLinear>& desc,
        const std::int64_t support_vector_count,
        const table& support_indices,
        const Float* expected_responses) {
        CAPTURE(support_vector_count);

        INFO("run training");
        auto train_result = this->train(desc, train_data, train_responses);
        const auto model = train_result.get_model();
        check_train_result(train_data, train_result, support_vector_count, support_indices);

        INFO("check if first and second class response is expected");
        REQUIRE(model.get_first_class_response() == expected_responses[0]);
        REQUIRE(model.get_second_class_response() == expected_responses[1]);
    }

    void check_weights(
        const table& train_data,
        const table& train_responses,
        const table& train_weights,
        const svm::descriptor<Float, Method, svm::task::classification, KernelTypeLinear>& desc,
        const std::int64_t support_vector_count,
        const table& support_indices,
        const table& test_data,
        const table& expected_decision_function) {
        CAPTURE(support_vector_count);

        INFO("run training");
        auto train_result = this->train(desc, train_data, train_responses, train_weights);
        const auto model = train_result.get_model();
        check_train_result(train_data, train_result, support_vector_count, support_indices);

        INFO("run inference");
        const auto infer_result = this->infer(desc, model, test_data);
        const auto decision_function = infer_result.get_decision_function();

        INFO("check if decision fuction is expected");
        check_table_match(expected_decision_function, decision_function);
    }

    void check_train_result(const table& train_data,
                            const svm::train_result<>& result,
                            const std::int64_t support_vector_count,
                            const table& support_indices) {
        check_shapes(train_data, result, support_vector_count);
        check_nans(result);

        INFO("check if support_indices values is expected");
        check_table_match(support_indices, result.get_support_indices());
    }

    void check_infer_result(const table& infer_data,
                            const svm::infer_result<>& result,
                            const table& decision_function,
                            const table& responses) {
        check_shapes(infer_data, result);
        check_nans(result);

        if (decision_function.has_data()) {
            INFO("check if decision_function values is expected");
            check_table_match(decision_function, result.get_decision_function());
        }

        INFO("check if responses values is expected");
        check_table_match(responses, result.get_responses());
    }

    void check_table_match(const table& reference, const table& actual_value) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::rel_error(reference, actual_value, tol);
        CHECK(diff < tol);
    }

    void check_shapes(const table& train_data,
                      const svm::train_result<>& result,
                      const std::int64_t support_vector_count) {
        const auto [support_vectors, support_indices, coeffs] = unpack_result(result);

        INFO("check if support_vector_count is expected");
        REQUIRE(result.get_support_vector_count() == support_vector_count);

        INFO("check if support_vectors shape is expected");
        REQUIRE(support_vectors.get_row_count() == support_vector_count);
        REQUIRE(support_vectors.get_column_count() == train_data.get_column_count());

        INFO("check if support_indices shape is expected");
        REQUIRE(support_indices.get_row_count() == support_vector_count);
        REQUIRE(support_indices.get_column_count() == 1);

        INFO("check if coeffs shape is expected");
        REQUIRE(coeffs.get_row_count() == support_vector_count);
        REQUIRE(coeffs.get_column_count() == 1);
    }

    void check_nans(const svm::train_result<>& result) {
        const auto [support_vectors, support_indices, coeffs] = unpack_result(result);

        INFO("check if there is no NaN in support_vectors");
        REQUIRE(te::has_no_nans(support_vectors));

        INFO("check if there is no NaN in support_indices");
        REQUIRE(te::has_no_nans(support_indices));

        INFO("check if there is no NaN in coeffs");
        REQUIRE(te::has_no_nans(coeffs));
    }

    void check_shapes(const table& infer_data, const svm::infer_result<>& result) {
        const auto [responses, decision_function] = unpack_result(result);

        INFO("check if responses shape is expected");
        REQUIRE(responses.get_row_count() == infer_data.get_row_count());
        REQUIRE(responses.get_column_count() == 1);

        INFO("check if decision_function shape is expected");
        REQUIRE(decision_function.get_row_count() == infer_data.get_row_count());
        REQUIRE(decision_function.get_column_count() == 1);
    }

    void check_nans(const svm::infer_result<>& result) {
        const auto [responses, decision_function] = unpack_result(result);

        INFO("check if there is no NaN in responses");
        REQUIRE(te::has_no_nans(responses));

        INFO("check if there is no NaN in decision_function");
        REQUIRE(te::has_no_nans(decision_function));
    }

    template <class KernelType>
    void check_kernel_accuracy(
        const table& train_data,
        const table& train_responses,
        const table& test_data,
        const table& test_responses,
        svm::descriptor<Float, Method, svm::task::classification, KernelType>& desc,
        const Float ref_accuracy) {
        INFO("set desctiptor parameters");
        desc.set_accuracy_threshold(0.001);
        desc.set_max_iteration_count(10 * train_data.get_row_count());
        desc.set_cache_size(2048.0);
        desc.set_tau(1.0e-6);

        INFO("run training");
        auto train_result = this->train(desc, train_data, train_responses);
        const auto model = train_result.get_model();
        check_shapes(train_data, train_result, model.get_support_vector_count());
        check_nans(train_result);

        INFO("run inference");
        const auto infer_result = this->infer(desc, model, test_data);
        check_shapes(test_data, infer_result);
        check_nans(infer_result);

        const Float tolerance = 1e-5;

        const auto score_table =
            te::accuracy_score<Float>(infer_result.get_responses(), test_responses, tolerance);
        const auto score = row_accessor<const Float>(score_table).pull({ 0, -1 })[0];

        CAPTURE(score);
        REQUIRE(score >= ref_accuracy);
    }

private:
    static auto unpack_result(const svm::train_result<>& result) {
        const auto support_vectors = result.get_support_vectors();
        const auto support_indices = result.get_support_indices();
        const auto coeffs = result.get_coeffs();
        return std::make_tuple(support_vectors, support_indices, coeffs);
    }

    static auto unpack_result(const svm::infer_result<>& result) {
        const auto responses = result.get_responses();
        const auto decision_function = result.get_decision_function();
        return std::make_tuple(responses, decision_function);
    }
};

using svm_types = COMBINE_TYPES((float, double), (svm::method::thunder, svm::method::smo));
using svm_nightly_types = COMBINE_TYPES((float, double), (svm::method::thunder));

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm polynomial manual dataset",
                     "[svm][integration][batch][polynomial]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->kernel_not_available_on_device());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = polynomial::descriptor<float_t, polynomial::method::dense>;

    constexpr std::int64_t row_count_train = 19;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<float_t, element_count_train> x_data = {
        -5, 2, -4, 1,  -3, 0, -2, -1, -1, -2, 0, -3, 1, -2, 2, -1, 3, 0, 4,
        1,  5, 2,  -1, 1,  0, 1,  1,  1,  -2, 2, -1, 2, 0,  2, 1,  2, 2, 2,
    };
    const auto x_train = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = { -1, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, 1,  1,  1,
                                                              1,  1,  1,  1,  1 };
    const auto y_train = homogen_table::wrap(y_data.data(), row_count_train, 1);

    constexpr std::int64_t row_count_test = 3;
    constexpr std::int64_t element_count_test = row_count_test * column_count;

    constexpr std::array<float_t, element_count_test> x_data_train = { 0, 0, -1, -1, 1, -1 };
    const auto x_test = homogen_table::wrap(x_data_train.data(), row_count_test, column_count);

    constexpr std::array<float_t, row_count_test> y_data_train = { 1, -1, -1 };
    const auto y_test = homogen_table::wrap(y_data_train.data(), row_count_test, 1);

    const double scale = 1;
    const double shift = 4;
    const double degree = 2;
    const double c = 1;

    const auto kernel_desc = kernel_t{}.set_scale(scale).set_shift(shift).set_degree(degree);
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    const double ref_accuracy = 1;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm sigmoid manual dataset",
                     "[svm][integration][batch][sigmoid]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->kernel_not_available_on_device());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = sigmoid::descriptor<float_t, sigmoid::method::dense>;

    constexpr std::int64_t row_count_train = 19;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<float_t, element_count_train> x_data = {
        -5, 2, -4, 1,  -3, 0, -2, -1, -1, -2, 0, -3, 1, -2, 2, -1, 3, 0, 4,
        1,  5, 2,  -1, 1,  0, 1,  1,  1,  -2, 2, -1, 2, 0,  2, 1,  2, 2, 2,
    };
    const auto x_train = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = { -1, -1, -1, -1, -1, -1, -1,
                                                              -1, -1, -1, -1, 1,  1,  1,
                                                              1,  1,  1,  1,  1 };
    const auto y_train = homogen_table::wrap(y_data.data(), row_count_train, 1);

    constexpr std::int64_t row_count_test = 3;
    constexpr std::int64_t element_count_test = row_count_test * column_count;

    constexpr std::array<float_t, element_count_test> x_data_train = { 0, 0, -1, -1, 1, -1 };
    const auto x_test = homogen_table::wrap(x_data_train.data(), row_count_test, column_count);

    constexpr std::array<float_t, row_count_test> y_data_train = { 1, -1, -1 };
    const auto y_test = homogen_table::wrap(y_data_train.data(), row_count_test, 1);

    const double scale = 1;
    const double shift = 0;
    const double c = 1;

    const auto kernel_desc = kernel_t{}.set_scale(scale).set_shift(shift);
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    const double ref_accuracy = 0.66666;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm polynomial mnist 2k",
                     "[svm][integration][batch][polynomial][external-dataset]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->kernel_not_available_on_device());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = polynomial::descriptor<float_t, polynomial::method::dense>;

    const te::dataframe train_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "svm/mnist_train_38_binary.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "svm/mnist_test_38_binary.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const auto kernel_desc = kernel_t{}.set_scale(3).set_shift(4).set_degree(3);

    const double c = 1.5e-3;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    const double ref_accuracy = 0.992;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify linear separable surface with big margin",
                     "[svm][integration][batch][linear]",
                     svm_types) {
    // TODO: Fix problem with incorrect number of support vectors on CPU
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<float_t, element_count_train> x_data = {
        -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = {
        -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    };
    const auto y = homogen_table::wrap(y_data.data(), row_count_train, 1);

    const double scale = 1.0;
    const double c = 1e-1;

    const auto kernel_desc = kernel_t{}.set_scale(scale);
    const auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    constexpr std::int64_t support_vector_count = 6;

    constexpr std::array<float_t, support_vector_count> support_indices_data = { 0, 1, 2, 3, 4, 5 };
    const auto support_indices =
        homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    constexpr std::array<float_t, row_count_train> decision_function_data = { -1.0, -2.0 / 3,
                                                                              -1.0, 2.0 / 3,
                                                                              1.0,  1.0 };
    const auto decision_function =
        homogen_table::wrap(decision_function_data.data(), row_count_train, 1);

    constexpr std::array<float_t, row_count_train> responses_data = {
        -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    };
    const auto responses = homogen_table::wrap(responses_data.data(), row_count_train, 1);

    this->check_kernel(x,
                       y,
                       svm_desc,
                       support_vector_count,
                       support_indices,
                       decision_function,
                       responses);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify linear not separable surface",
                     "[svm][integration][batch][linear]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    constexpr std::int64_t row_count_train = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<float_t, element_count_train> x_data = {
        -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, -3.0, -3.0, 3.0, 3.0
    };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = { -1.0, -1.0, -1.0, 1.0,
                                                              1.0,  1.0,  1.0,  -1.0 };
    const auto y = homogen_table::wrap(y_data.data(), row_count_train, 1);

    const double scale = 1.0;
    const double c = 1.0;

    const auto kernel_desc = kernel_t{}.set_scale(scale);
    const auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    constexpr std::int64_t support_vector_count = 8;

    constexpr std::array<float_t, support_vector_count> support_indices_data = { 0, 1, 2, 3,
                                                                                 4, 5, 6, 7 };
    const auto support_indices =
        homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    constexpr std::array<float_t, row_count_train> decision_function_data = {
        -1.0, -2.0 / 3, -1.0, 2.0 / 3, 1.0, 1.0, -2.0, 2.0
    };
    const auto decision_function =
        homogen_table::wrap(decision_function_data.data(), row_count_train, 1);

    constexpr std::array<float_t, row_count_train> responses_data = { -1.0, -1.0, -1.0, 1.0,
                                                                      1.0,  1.0,  -1.0, 1.0 };
    const auto responses = homogen_table::wrap(responses_data.data(), row_count_train, 1);

    this->check_kernel(x,
                       y,
                       svm_desc,
                       support_vector_count,
                       support_indices,
                       decision_function,
                       responses);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify quadric separable surface with rbf kernel",
                     "[svm][integration][batch][rbf]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    constexpr std::int64_t row_count_train = 12;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<float_t, element_count_train> x_data = {
        -2.0, 0.0, -2.0, -1.0, -2.0, 1.0, 2.0, 0.0, 2.0, -1.0, 2.0, 1.0,
        -1.0, 0.0, -1.0, -0.5, -1.0, 0.5, 1.0, 0.5, 1.0, -0.5, 1.0, 0.5
    };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                              1.0,  1.0,  1.0,  1.0,  1.0,  1.0 };
    const auto y = homogen_table::wrap(y_data.data(), row_count_train, 1);

    const double sigma = 1.0;
    const double c = 1.0;

    const auto kernel_desc = kernel_t{}.set_sigma(sigma);
    const auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    constexpr std::int64_t support_vector_count = 12;

    constexpr std::array<float_t, support_vector_count> support_indices_data = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    };
    const auto support_indices =
        homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    this->check_kernel(x, y, svm_desc, support_vector_count, support_indices, homogen_table{}, y);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify any two responses",
                     "[svm][integration][batch][linear]",
                     svm_types) {
    // TODO: Fix problem with incorrect number of support vectors on CPU
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    using responses_pair_t =
        std::pair<std::array<float_t, row_count_train>, std::array<float_t, 2>>;

    constexpr std::array<float_t, element_count_train> x_data = { -2.0, -1.0, -1.0, -1.0,
                                                                  -1.0, -2.0, 1.0,  1.0,
                                                                  1.0,  2.0,  2.0,  1.0 };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    const auto [responses_data, unique_responses] = GENERATE_COPY(
        responses_pair_t({ -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 }, { -1.0, 1.0 }),
        responses_pair_t({ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 }, { 0.0, 1.0 }),
        responses_pair_t({ 0.0, 0.0, 0.0, 2.0, 2.0, 2.0 }, { 0.0, 2.0 }),
        responses_pair_t({ -1.0, -1.0, -1.0, 0.0, 0.0, 0.0 }, { -1.0, 0.0 }));

    const auto y = homogen_table::wrap(responses_data.data(), row_count_train, 1);

    const double scale = 1.0;
    const double c = 1e-1;

    const auto kernel_desc = kernel_t{}.set_scale(scale);
    const auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    constexpr std::int64_t support_vector_count = 6;

    constexpr std::array<float_t, support_vector_count> support_indices_data = { 0, 1, 2, 3, 4, 5 };
    const auto support_indices =
        homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    this->check_different_responses(x,
                                    y,
                                    svm_desc,
                                    support_vector_count,
                                    support_indices,
                                    unique_responses.data());
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify linear separable surface with weights",
                     "[svm][integration][batch][linear]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->weights_not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    using weights_pair_t = std::pair<std::array<float_t, row_count_train>, std::array<float_t, 1>>;

    constexpr std::array<float_t, element_count_train> x_data = { -2.0, 0.0, -1.0, -1.0, 0.0, -2.0,
                                                                  0.0,  2.0, 1.0,  1.0,  2.0, 0.0 };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<float_t, row_count_train> y_data = {
        -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    };
    const auto y = homogen_table::wrap(y_data.data(), row_count_train, 1);

    const auto [weights_data_array, decision_function_data] = GENERATE_COPY(
        weights_pair_t({ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 0.0 }),
        weights_pair_t({ 10.0, 0.1, 0.1, 0.1, 0.1, 10.0 }, { -0.44 }),
        weights_pair_t({ 0.1, 0.1, 10.0, 10.0, 0.1, 0.1 }, { 0.44 }));

    const auto weights = homogen_table::wrap(weights_data_array.data(), row_count_train, 1);

    constexpr std::array<float_t, 2> x_test_data = { -1.0, 1.0 };
    const auto x_test = homogen_table::wrap(x_test_data.data(), 1, column_count);

    const double scale = 1.0;
    const double c = 1e-1;

    const auto kernel_desc = kernel_t{}.set_scale(scale);
    const auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    constexpr std::int64_t support_vector_count = 6;

    constexpr std::array<float_t, support_vector_count> support_indices_data = { 0, 1, 2, 3, 4, 5 };
    const auto support_indices =
        homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    const auto decision_function = homogen_table::wrap(decision_function_data.data(), 1, 1);

    this->check_weights(x,
                        y,
                        weights,
                        svm_desc,
                        support_vector_count,
                        support_indices,
                        x_test,
                        decision_function);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm linear gisette 6k x 5k",
                     "[svm][integration][batch][linear][external-dataset]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/gisette/dataset/gisette_train_6k.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/gisette/dataset/gisette_test_1k.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const double c = 1.5e-3;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.975;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm rbf covertype 100k x 54",
                     "[svm][integration][batch][rbf][external-dataset]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/covertype/dataset/covertype_binary_train_100k.csv" });
    const auto feature_count = train_data.get_column_count();
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train =
        train_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/covertype/dataset/covertype_binary_test_100k.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test =
        test_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const auto kernel_desc = kernel_t{}.set_sigma(std::sqrt(feature_count) * 2.0);

    const double c = 1.0e3;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{ kernel_desc }
            .set_c(c);

    const double ref_accuracy = 0.9878;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm rbf epsilon 16k x 2k",
                     "[svm][integration][batch][rbf][nightly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_16k_train.csv" });
    const auto feature_count = train_data.get_column_count();
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train =
        train_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_16k_test.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test =
        test_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const auto kernel_desc = kernel_t{}.set_sigma(std::sqrt(feature_count) * 2.0);

    const double c = 9.0e2;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.8538;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm linear higgs 100k x 28",
                     "[svm][integration][batch][linear][nightly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    const te::dataframe train_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_100t_train.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_50t_test.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const double c = 1.0;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.6395;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm linear cifar 50k x 3072",
                     "[svm][integration][batch][linear][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/cifar/dataset/cifar_50k_train_binary.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/cifar/dataset/cifar_10k_test_binary.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const double c = 1.0e-7;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.9;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm rbf imdb_drama 121k x 1001",
                     "[svm][integration][batch][rbf][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/imdb_drama/dataset/imdb_drama_120k.csv" });
    const auto feature_count = train_data.get_column_count();
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train =
        train_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/imdb_drama/dataset/imdb_drama_120k.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test =
        test_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const auto kernel_desc = kernel_t{}.set_sigma(std::sqrt(feature_count) * 2.0);

    const double c = 50;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.6379;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm rbf epsilon 50k x 2k",
                     "[svm][integration][batch][rbf][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_50k_train.csv" });
    const auto feature_count = train_data.get_column_count();
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train =
        train_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_50k_train.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test =
        test_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const auto kernel_desc = kernel_t{}.set_sigma(std::sqrt(feature_count) * 2.0);

    const double c = 1.0e3;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.8842;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm linear epsilon 80k x 2k",
                     "[svm][integration][batch][linear][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_80k_train.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_80k_train.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const double c = 1.0;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.9025;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm rbf cifar 50k x 3072",
                     "[svm][integration][batch][rbf][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = rbf::descriptor<float_t, rbf::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/cifar/dataset/cifar_50k_train_binary.csv" });
    const auto feature_count = train_data.get_column_count();
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train =
        train_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/cifar/dataset/cifar_10k_test_binary.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test =
        test_data.get_table(this->get_homogen_table_id(), range(feature_count - 1, feature_count));

    const auto kernel_desc = kernel_t{}.set_sigma(std::sqrt(feature_count) * 2.0);

    const double c = 1.0e-5;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.8999;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm linear imdb_drama 121k x 1001",
                     "[svm][integration][batch][linear][weekly][external-dataset]",
                     svm_nightly_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using kernel_t = linear::descriptor<float_t, linear::method::dense>;

    const te::dataframe train_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/imdb_drama/dataset/imdb_drama_120k.csv" });
    const auto x_train = train_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const auto y_train = train_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const te::dataframe test_data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/imdb_drama/dataset/imdb_drama_120k.csv" });
    const table x_test = test_data.get_table(this->get_homogen_table_id(), range(0, -1));
    const table y_test = test_data.get_table(
        this->get_homogen_table_id(),
        range(train_data.get_column_count() - 1, train_data.get_column_count()));

    const double c = 1.0e-3;
    auto svm_desc =
        svm::descriptor<float_t, method_t, svm::task::classification, kernel_t>{}.set_c(c);

    const double ref_accuracy = 0.6379;

    this->check_kernel_accuracy(x_train, y_train, x_test, y_test, svm_desc, ref_accuracy);
}

} // namespace oneapi::dal::svm::test
