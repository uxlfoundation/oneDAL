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

#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using distance_types = std::tuple<float, double>;

template <typename Float>
class sql2_distance_test_random : public te::float_algo_fixture<Float> {
public:
    void generate() {
        r_count1_ = GENERATE(17, 31);
        r_count2_ = GENERATE(7, 29);
        c_count_ = GENERATE(3, 13);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output() {
        return ndarray<Float, 2>::zeros(this->get_queue(), { r_count1_, r_count2_ });
    }

    void generate_input() {
        const auto input1_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ r_count1_, c_count_ }.fill_uniform(-0.2, 0.5));
        this->input_table1_ = input1_dataframe.get_table(this->get_homogen_table_id());
        const auto input2_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ r_count2_, c_count_ }.fill_uniform(-0.5, 1.0));
        this->input_table2_ = input2_dataframe.get_table(this->get_homogen_table_id());
    }

    void squared_l2_groundtruth_check(const ndview<Float, 2>& out, const Float atol = 1.e-3) {
        for (std::int64_t i = 0; i < r_count1_; ++i) {
            const auto inp_row1 =
                row_accessor<const Float>{ input_table1_ }.pull(this->get_queue(), { i, i + 1 });
            for (std::int64_t j = 0; j < r_count2_; ++j) {
                const auto inp_row2 =
                    row_accessor<const Float>{ input_table2_ }.pull(this->get_queue(),
                                                                    { j, j + 1 });
                Float gtv = 0;
                for (std::int64_t k = 0; k < c_count_; ++k) {
                    const auto diff = inp_row1[k] - inp_row2[k];
                    gtv += diff * diff;
                }
                const auto val = *(out.get_data() + out.get_leading_stride() * i + j);
                const auto diff = gtv - val;
                CAPTURE(gtv, val, i, j, r_count1_, r_count2_, c_count_);
                REQUIRE(-atol <= diff);
                CAPTURE(gtv, val, i, j, r_count1_, r_count2_, c_count_);
                REQUIRE(diff <= atol);
            }
        }
    }

    void test_squared_l2_distance() {
        auto input1_arr = row_accessor<const Float>{ input_table1_ }.pull(this->get_queue());
        auto input2_arr = row_accessor<const Float>{ input_table2_ }.pull(this->get_queue());
        auto input1 = ndview<Float, 2>::wrap(input1_arr.get_data(), { r_count1_, c_count_ });
        auto input2 = ndview<Float, 2>::wrap(input2_arr.get_data(), { r_count2_, c_count_ });
        auto [output, output_event] = this->output();
        distance<Float, squared_l2_metric<Float>> sql2_distance(this->get_queue());
        auto distance_event = sql2_distance(input1, input2, output, { output_event });
        distance_event.wait_and_throw();
        squared_l2_groundtruth_check(output);
    }

private:
    table input_table1_;
    table input_table2_;
    std::int64_t c_count_;
    std::int64_t r_count1_;
    std::int64_t r_count2_;
};

TEMPLATE_LIST_TEST_M(sql2_distance_test_random,
                     "Randomly filled squared L2-distance computation",
                     "[l2][distance][small]",
                     distance_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_squared_l2_distance();
}

} // namespace oneapi::dal::backend::primitives::test
