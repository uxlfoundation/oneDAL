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

#include "oneapi/dal/algo/linear_kernel/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::linear_kernel::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class linear_kernel_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(double scale, double shift) const {
        return linear_kernel::descriptor<Float, Method>{}.set_scale(scale).set_shift(shift);
    }

    void general_checks(const te::dataframe& x_data,
                        const te::dataframe& y_data,
                        double scale,
                        double shift,
                        const te::table_id& x_data_table_id,
                        const te::table_id& y_data_table_id) {
        CAPTURE(scale);
        CAPTURE(shift);
        const table x = this->get_device_table(x_data, x_data_table_id);
        const table y = this->get_device_table(y_data, y_data_table_id);

        INFO("create descriptor");
        const auto linear_kernel_desc = get_descriptor(scale, shift);

        INFO("run compute");
        const auto compute_result = this->compute(linear_kernel_desc, x, y);
        check_compute_result(scale, shift, x, y, compute_result);
    }

    void check_compute_result(double scale,
                              double shift,
                              const table& x_data,
                              const table& y_data,
                              const linear_kernel::compute_result<>& result) {
        const auto result_values = result.get_values();

        INFO("check if result values table shape is expected");
        REQUIRE(result_values.get_row_count() == x_data.get_row_count());
        REQUIRE(result_values.get_column_count() == y_data.get_row_count());

        INFO("check if there is no NaN in result values table");
        REQUIRE(te::has_no_nans(result_values));

        INFO("check if result values are expected");
        check_result_values(scale, shift, x_data, y_data, result_values);
    }

    void check_result_values(double scale,
                             double shift,
                             const table& x_data,
                             const table& y_data,
                             const table& result_values) {
        const auto reference = compute_reference(scale, shift, x_data, y_data);
        const double tol = te::get_tolerance<Float>(3e-4, 1e-9);
        const double diff = te::abs_error(reference, result_values);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference(double scale,
                                         double shift,
                                         const table& x_data,
                                         const table& y_data) {
        const auto x_data_matrix = la::matrix<double>::wrap(x_data);
        const auto y_data_matrix = la::matrix<double>::wrap(y_data);

        auto reference = la::dot(x_data_matrix, y_data_matrix.t(), scale);

        la::enumerate_linear(reference, [&](std::int64_t i, double) {
            reference.set(i) += shift;
        });

        return reference;
    }
};

using linear_kernel_types = COMBINE_TYPES((float, double), (linear_kernel::method::dense));

TEMPLATE_LIST_TEST_M(linear_kernel_batch_test,
                     "linear_kernel common flow",
                     "[linear_kernel][integration][batch]",
                     linear_kernel_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 250, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1100, 50 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 8888),
                           te::dataframe_builder{ 200, 50 }.fill_normal(0, 1, 8888),
                           te::dataframe_builder{ 1000, 50 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double scale = GENERATE_COPY(1.0, 2.0);
    const double shift = GENERATE_COPY(0.0, 1.0);

    this->general_checks(x_data, y_data, scale, shift, x_data_table_id, y_data_table_id);
}

TEMPLATE_LIST_TEST_M(linear_kernel_batch_test,
                     "linear_kernel compute one element matrix",
                     "[linear_kernel][integration][batch]",
                     linear_kernel_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double scale = GENERATE_COPY(1.0, 2.0);
    const double shift = GENERATE_COPY(0.0, 1.0);

    this->general_checks(x_data, y_data, scale, shift, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::linear_kernel::test
