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
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace cov = oneapi::dal::covariance;

template <typename TestType>
class covariance_spmd_test : public covariance_test<TestType, covariance_spmd_test<TestType>> {
public:
    using base_t = covariance_test<TestType, covariance_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::input_t;
    using result_t = typename base_t::result_t;
    using descriptor_t = typename base_t::descriptor_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    result_t compute_override(Args&&... args) {
        return this->compute_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    result_t merge_compute_result_override(const std::vector<result_t>& results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<input_t> split_compute_input_override(std::int64_t split_count, Args&&... args) {
        const input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                input_t{ split_data[i] });
        }

        return split_input;
    }

    void spmd_general_checks(const te::dataframe& data_fr,
                             const te::table_id& data_table_id,
                             descriptor_t cov_desc) {
        const table data = data_fr.get_table(this->get_policy(), data_table_id);

        const auto compute_result = this->compute_override(cov_desc, data);

        base_t::check_compute_result(cov_desc, data, compute_result);
    }

private:
    std::int64_t rank_count_;
};

using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

TEMPLATE_LIST_TEST_M(covariance_spmd_test,
                     "covariance common flow",
                     "[covariance][integration][spmd]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const int rank_count = GENERATE(2, 4);
    INFO("rank_count=" << rank_count);
    this->set_rank_count(rank_count);

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

    auto cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                        .set_result_options(result_option)
                        .set_assume_centered(assume_centered)
                        .set_bias(bias);

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 10000, 200 }.fill_uniform(-30, 30, 7777));

    INFO("num_rows=" << input.get_row_count());
    INFO("num_columns=" << input.get_column_count());

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();

    this->spmd_general_checks(input, input_data_table_id, cov_desc);
}

} // namespace oneapi::dal::covariance::test
