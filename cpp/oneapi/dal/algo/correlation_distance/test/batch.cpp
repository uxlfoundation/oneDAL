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

#include "oneapi/dal/algo/correlation_distance/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::correlation_distance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class correlation_distance_batch_test
        : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor() const {
        return correlation_distance::descriptor<Float, Method>{};
    }

    void general_checks(const te::dataframe& x_data,
                        const te::dataframe& y_data,
                        const te::table_id& x_data_table_id,
                        const te::table_id& y_data_table_id) {
        const table x = x_data.get_table(this->get_policy(), x_data_table_id);
        const table y = y_data.get_table(this->get_policy(), y_data_table_id);

        INFO("create descriptor");
        const auto correlation_distance_desc = get_descriptor();

        INFO("run compute");
        const auto compute_result = this->compute(correlation_distance_desc, x, y);
        check_compute_result(x, y, compute_result);
    }

    void check_compute_result(const table& x_data,
                              const table& y_data,
                              const correlation_distance::compute_result<>& result) {
        const auto result_values = result.get_values();

        INFO("check if result values table shape is expected");
        REQUIRE(result_values.get_row_count() == x_data.get_row_count());
        REQUIRE(result_values.get_column_count() == y_data.get_row_count());

        INFO("check if there is no NaN in result values table");
        REQUIRE(te::has_no_nans(result_values));

        INFO("check if result values are expected");
        check_result_values(x_data, y_data, result_values);
    }

    void check_result_values(const table& x_data, const table& y_data, const table& result_values) {
        const auto reference = compute_reference(x_data, y_data);

        const auto col_count = reference.get_column_count();
        const auto row_count = reference.get_row_count();
        REQUIRE(row_count == result_values.get_row_count());
        REQUIRE(col_count == result_values.get_column_count());

        row_accessor<const Float> acc{ result_values };
        for (std::int64_t row = 0; row < row_count; ++row) {
            auto row_arr = acc.pull({ row, row + 1 });
            for (std::int64_t col = 0; col < col_count; ++col) {
                const Float res = row_arr[col];
                const Float gtr = reference.get(row, col);
                const auto rerr = std::abs(res - gtr) /
                                  std::max<double>({ double(1), std::abs(res), std::abs(gtr) });
                CAPTURE(row_count, col_count, x_data.get_column_count(), row, col, res, gtr, rerr);
                if (rerr > 1e-4)
                    FAIL();
            }
        }
    }

    la::matrix<double> compute_reference(const table& x_data, const table& y_data) {
        const auto x_data_matrix = la::matrix<double>::wrap(x_data);
        const auto y_data_matrix = la::matrix<double>::wrap(y_data);
        const auto row_count_x = x_data_matrix.get_row_count();
        const auto row_count_y = y_data_matrix.get_row_count();
        const auto column_count = x_data_matrix.get_column_count();
        auto reference = la::matrix<double>::full({ row_count_x, row_count_y }, 0.0);

        // For each pair of vectors
        for (std::int64_t i = 0; i < row_count_x; ++i) {
            for (std::int64_t j = 0; j < row_count_y; ++j) {
                // Calculate sum of current row
                Float mean_x = 0.0, mean_y = 0.0;
                for (std::int64_t k = 0; k < column_count; ++k) {
                    mean_x += x_data_matrix.get(i, k);
                    mean_y += y_data_matrix.get(j, k);
                }

                // Calculate mean
                mean_x /= column_count;
                mean_y /= column_count;

                // Calculate the numerator (covariance)
                Float numerator = 0.0, x_variance = 0.0, y_variance = 0.0;
                for (std::int64_t k = 0; k < column_count; ++k) {
                    const Float x_centered = x_data_matrix.get(i, k) - mean_x;
                    const Float y_centered = y_data_matrix.get(j, k) - mean_y;

                    x_variance += x_centered * x_centered;
                    y_variance += y_centered * y_centered;
                    numerator += x_centered * y_centered;
                }

                // Calculate correlation coefficient
                Float denominator = std::sqrt(x_variance) * std::sqrt(y_variance);

                if (denominator > 0.0) {
                    reference.set(i, j) = Float(1.0) - (numerator / denominator);
                }
                else {
                    reference.set(i, j) = Float(1.0);
                }
            }
        }
        return reference;
    }
};

using correlation_distance_types = COMBINE_TYPES((float, double),
                                                 (correlation_distance::method::dense));

TEMPLATE_LIST_TEST_M(correlation_distance_batch_test,
                     "correlation_distance common flow",
                     "[correlation_distance][integration][batch]",
                     correlation_distance_types) {
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

    this->general_checks(x_data, y_data, x_data_table_id, y_data_table_id);
}

TEMPLATE_LIST_TEST_M(correlation_distance_batch_test,
                     "correlation_distance compute one element matrix",
                     "[correlation_distance][integration][batch]",
                     correlation_distance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    this->general_checks(x_data, y_data, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::correlation_distance::test
