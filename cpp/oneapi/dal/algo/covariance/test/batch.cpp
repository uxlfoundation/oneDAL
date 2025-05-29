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

#include "oneapi/dal/algo/covariance/test/fixture.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace cov = oneapi::dal::covariance;

template <typename TestType>
class covariance_batch_test : public covariance_test<TestType, covariance_batch_test<TestType>> {
public:
    using base_t = covariance_test<TestType, covariance_batch_test<TestType>>;
    using descriptor_t = typename base_t::descriptor_t;

    void general_checks(const te::dataframe& input,
                        const te::table_id& input_table_id,
                        descriptor_t cov_desc,
                        const bool non_batched) {
        const table data = input.get_table(this->get_policy(), input_table_id);

        detail::compute_parameters parameters{};
        parameters.set_cpu_max_cols_batched(1);
        auto compute_result =
            non_batched ? this->compute(cov_desc, parameters, data) : this->compute(cov_desc, data);
        this->check_compute_result(cov_desc, data, compute_result);
    }
};

TEMPLATE_LIST_TEST_M(covariance_batch_test,
                     "covariance common flow",
                     "[covariance][integration][online]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const bool assume_centered = GENERATE(true, false);
    INFO("assume_centered=" << assume_centered);
    const bool bias = GENERATE(true, false);
    INFO("bias=" << bias);
    const cov::result_option_id result_option =
        GENERATE(covariance::result_options::means,
                 covariance::result_options::cov_matrix,
                 covariance::result_options::cor_matrix,
                 covariance::result_options::cor_matrix | covariance::result_options::cov_matrix,
                 covariance::result_options::cor_matrix | covariance::result_options::cov_matrix |
                     covariance::result_options::means);
    INFO("result_option=" << result_option);

    const bool non_batched = GENERATE(false, true);
    INFO("non_batched=" << non_batched);

    auto cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                        .set_result_options(result_option)
                        .set_assume_centered(assume_centered)
                        .set_bias(bias);

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 500, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1000, 20 }.fill_uniform(-30, 30, 7777));

    INFO("num_rows=" << input.get_row_count());
    INFO("num_columns=" << input.get_column_count());

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->general_checks(input, input_data_table_id, cov_desc, non_batched);
}

} // namespace oneapi::dal::covariance::test
