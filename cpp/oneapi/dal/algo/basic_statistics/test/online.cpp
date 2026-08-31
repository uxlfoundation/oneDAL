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
