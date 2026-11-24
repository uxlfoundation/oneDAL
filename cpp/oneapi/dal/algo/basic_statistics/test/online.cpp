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

#include <utility>
#include <vector>

#include "oneapi/dal/algo/basic_statistics/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;

template <typename TestType>
class basic_statistics_online_test
        : public basic_statistics_test<TestType, basic_statistics_online_test<TestType>> {};

TEMPLATE_LIST_TEST_M(basic_statistics_online_test,
                     "basic_statistics common flow no weights",
                     "[basic_statistics][integration][online]",
                     basic_statistics_types) {
    SKIP_IF(this->not_float64_friendly());
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 6000, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 6000, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 20 }.fill_normal(-0.5, 0.5, 7777));

    std::shared_ptr<te::dataframe> weights;
    const bool use_weights = GENERATE(0, 1);
    const int64_t nBlocks = GENERATE(1, 3, 10);

    if (use_weights) {
        const auto row_count = data.get_row_count();
        weights = std::make_shared<te::dataframe>(
            te::dataframe_builder{ row_count, 1 }.fill_normal(0, 1, 777).build());
    }

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    this->online_general_checks(data, weights, compute_mode, nBlocks);
}

// A one-row block used to make the online variance NaN.
TEMPLATE_LIST_TEST_M(basic_statistics_online_test,
                     "basic_statistics online with one-row blocks",
                     "[basic_statistics][integration][online]",
                     basic_statistics_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    const float_t data[] = { 1, 2, 3, 6 };
    const auto desc =
        this->get_descriptor(result_options::sum_squares_centered | result_options::variance);

    bs::partial_compute_result<> partial;
    for (std::int64_t row = 0; row < 2; ++row) {
        partial = this->partial_compute(desc, partial, homogen_table::wrap(data + 2 * row, 1, 2));
    }
    const auto result = this->finalize_compute(desc, partial);

    const auto ssc = row_accessor<const float_t>(result.get_sum_squares_centered()).pull();
    const auto variance = row_accessor<const float_t>(result.get_variance()).pull();
    REQUIRE(ssc[0] == 2);
    REQUIRE(ssc[1] == 8);
    REQUIRE(variance[0] == 2);
    REQUIRE(variance[1] == 8);
}

// Means near 1e8 make m1*m1 + m2*m2 - 2*m1*m2 cancel; (m2 - m1)^2 keeps the variance.
TEMPLATE_LIST_TEST_M(basic_statistics_online_test,
                     "basic_statistics online merge with a large offset",
                     "[basic_statistics][integration][online]",
                     basic_statistics_types) {
    using float_t = std::tuple_element_t<0, TestType>;
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF((!std::is_same_v<float_t, double>));
    constexpr std::int64_t row_count = 3000;
    const te::dataframe df =
        te::dataframe_builder{ row_count, 1 }.fill_uniform(1e8 - 1.0, 1e8 + 1.0, 7777).build();
    const auto data = df.get_table(this->get_policy(), this->get_homogen_table_id());
    const auto mode = result_options::sum_squares_centered | result_options::variance;
    const auto desc = this->get_descriptor(mode);
    bs::partial_compute_result<> partial;
    for (const auto& block : te::split_table_by_rows<float_t>(this->get_policy(), data, 3)) {
        partial = this->partial_compute(desc, partial, block);
    }
    this->check_compute_result(mode, data, table{}, this->finalize_compute(desc, partial), 1e-6);
}

// finalize_compute must leave the partial result as it was, so it can be finalized again
// or extended with more blocks.
TEMPLATE_LIST_TEST_M(basic_statistics_online_test,
                     "basic_statistics finalize twice on the same partial result",
                     "[basic_statistics][integration][online]",
                     basic_statistics_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    const te::dataframe data = te::dataframe_builder{ 6, 3 }.fill_normal(-30, 30, 7777).build();
    const auto data_table = data.get_table(this->get_policy(), this->get_homogen_table_id());
    const auto desc =
        this->get_descriptor(bs::result_option_id(dal::result_option_id_base(mask_full)));
    const int64_t nBlocks = GENERATE(1, 3);

    bs::partial_compute_result<> partial;
    for (const auto& block :
         te::split_table_by_rows<float_t>(this->get_policy(), data_table, nBlocks)) {
        partial = this->partial_compute(desc, partial, block);
    }

    const std::vector<table> partial_tables = {
        partial.get_partial_n_rows(),      partial.get_partial_min(),
        partial.get_partial_max(),         partial.get_partial_sum(),
        partial.get_partial_sum_squares(), partial.get_partial_sum_squares_centered()
    };
    // A copy, not a view: for a host table row_accessor can return the table's own data.
    const auto copy_values = [](const table& t) {
        const auto ary = row_accessor<const float_t>{ t }.pull();
        return std::vector<float_t>(ary.get_data(), ary.get_data() + ary.get_count());
    };
    std::vector<std::vector<float_t>> before;
    for (const auto& t : partial_tables) {
        before.push_back(copy_values(t));
    }

    const auto first = this->finalize_compute(desc, partial);
    for (std::size_t i = 0; i < partial_tables.size(); ++i) {
        CAPTURE(i);
        REQUIRE(copy_values(partial_tables[i]) == before[i]);
    }

    const auto second = this->finalize_compute(desc, partial);
    const std::vector<std::pair<table, table>> results = {
        { first.get_min(), second.get_min() },
        { first.get_max(), second.get_max() },
        { first.get_sum(), second.get_sum() },
        { first.get_sum_squares(), second.get_sum_squares() },
        { first.get_sum_squares_centered(), second.get_sum_squares_centered() },
        { first.get_mean(), second.get_mean() },
        { first.get_second_order_raw_moment(), second.get_second_order_raw_moment() },
        { first.get_variance(), second.get_variance() },
        { first.get_standard_deviation(), second.get_standard_deviation() },
        { first.get_variation(), second.get_variation() },
    };
    for (std::size_t i = 0; i < results.size(); ++i) {
        CAPTURE(i);
        REQUIRE(copy_values(results[i].second) == copy_values(results[i].first));
    }
}

/// Sweep the result option masks that the common flow above does not reach.
///
/// The partial descriptor derived by `get_desc_to_compute` must request a superset
/// of what the user asked for: any statistic left out of it is never written by the
/// init step of `partial_compute`, so `finalize_compute` reads an empty partial
/// table. Mixed masks such as `min | max | sum` used to fall into a branch that
/// dropped min/max from the partial descriptor, which aborted the GPU online flow
/// with "Row count and column count do not match element count in array". The three
/// masks exercised by the common flow (`min | max`, `mean | variance`, all) all
/// happen to avoid that branch, so nothing caught it.
TEMPLATE_LIST_TEST_M(basic_statistics_online_test,
                     "basic_statistics online result option mask sweep",
                     "[basic_statistics][integration][online]",
                     basic_statistics_types) {
    SKIP_IF(this->not_float64_friendly());
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777));

    const std::int64_t nBlocks = GENERATE(1, 3);

    // Every single statistic on its own, plus the mixed masks that cross the
    // min/max and sums groups.
    const bs::result_option_id compute_mode =
        GENERATE_COPY(bs::result_option_id(result_options::min),
                      bs::result_option_id(result_options::max),
                      bs::result_option_id(result_options::sum),
                      bs::result_option_id(result_options::sum_squares),
                      bs::result_option_id(result_options::sum_squares_centered),
                      bs::result_option_id(result_options::mean),
                      bs::result_option_id(result_options::second_order_raw_moment),
                      bs::result_option_id(result_options::variance),
                      bs::result_option_id(result_options::standard_deviation),
                      bs::result_option_id(result_options::variation),
                      result_options::min | result_options::sum,
                      result_options::max | result_options::sum_squares,
                      result_options::min | result_options::max | result_options::sum,
                      result_options::min | result_options::max | result_options::sum_squares,
                      result_options::min | result_options::sum_squares_centered,
                      result_options::max | result_options::variance,
                      result_options::min | result_options::max | result_options::mean);

    this->online_general_checks(data, std::shared_ptr<te::dataframe>{}, compute_mode, nBlocks);
}

} // namespace oneapi::dal::basic_statistics::test
