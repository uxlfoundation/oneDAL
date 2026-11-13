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
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using types_to_check = std::tuple<float, double>;

template <typename Float>
class compute_squared_l2_norms_test_random : public te::float_algo_fixture<Float> {
public:
    void generate() {
        r_count_ = GENERATE(17, 31);
        c_count_ = GENERATE(3, 13);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output() {
        return ndarray<Float, 1>::zeros(this->get_queue(), { r_count_ });
    }

    void generate_input() {
        const auto input_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ r_count_, c_count_ }.fill_uniform(-0.2, 0.5));
        this->input_table_ = input_dataframe.get_table(this->get_homogen_table_id());
    }

    void squared_l2_norms_check(const ndview<Float, 1>& out, const Float atol = 1.e-3) {
        for (std::int64_t i = 0; i < r_count_; ++i) {
            const auto inp_row =
                row_accessor<const Float>{ input_table_ }.pull(this->get_queue(), { i, i + 1 });
            Float gtv = 0;
            for (std::int64_t j = 0; j < c_count_; ++j) {
                gtv += (inp_row[j] * inp_row[j]);
            }
            const Float val = *(out.get_data() + i);
            const Float diff = gtv - val;
            CAPTURE(gtv, val, i, r_count_, c_count_);
            REQUIRE(-atol <= diff);
            CAPTURE(gtv, val, i, r_count_, c_count_);
            REQUIRE(diff <= atol);
        }
    }

    void test_squared_l2_norms() {
        auto input_arr = row_accessor<const Float>{ input_table_ }.pull(this->get_queue());
        const auto input = ndview<Float, 2>::wrap(input_arr.get_data(), { r_count_, c_count_ });
        auto [output_arr, norms_event] =
            compute_squared_l2_norms(this->get_queue(), input, {}, sycl::usm::alloc::shared);
        norms_event.wait_and_throw();
        squared_l2_norms_check(output_arr);
    }

private:
    table input_table_;
    std::int64_t c_count_;
    std::int64_t r_count_;
};

TEMPLATE_LIST_TEST_M(compute_squared_l2_norms_test_random,
                     "Randomly filled squared L2-norms computation",
                     "[l2][norms][aux][small]",
                     types_to_check) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_squared_l2_norms();
}

template <typename Float>
class scatter_2d_test : public te::float_algo_fixture<Float> {
public:
    void generate() {
        count1_ = GENERATE(17, 31, 331);
        count2_ = GENERATE(3, 13, 337);
    }

    auto output() {
        return ndarray<Float, 2>::zeros(this->get_queue(), { count1_, count2_ });
    }

    auto generate_input(std::int64_t size, Float factor = 1.0) {
        auto res_array = ndarray<Float, 1>::empty(this->get_queue(), { size });
        auto* out_ptr = res_array.get_mutable_data();
        auto res_event = this->get_queue().parallel_for(size, [=](sycl::id<1> idx) {
            out_ptr[idx] = factor * Float(idx);
        });
        return std::make_tuple(res_array, res_event);
    }

    void scatter_check(const ndview<Float, 2>& out, const Float atol = 1.e-3) {
        for (std::int64_t i = 0; i < count1_; ++i) {
            for (std::int64_t j = 0; j < count2_; ++j) {
                const auto val = *(out.get_data() + out.get_leading_stride() * i + j);
                const auto gtv = Float(i) - Float(j);
                const auto diff = gtv - val;
                CAPTURE(gtv, val, i, j, count1_, count2_);
                REQUIRE(-atol <= diff);
                CAPTURE(gtv, val, i, j, count1_, count2_);
                REQUIRE(diff <= atol);
            }
        }
    }

    void test_scatter() {
        auto [input1_arr, input1_event] = generate_input(count1_, +1.0);
        auto [input2_arr, input2_event] = generate_input(count2_, -1.0);
        auto [output_arr, output_event] = output();
        auto scatter_event = scatter_2d(this->get_queue(),
                                        input1_arr,
                                        input2_arr,
                                        output_arr,
                                        { input1_event, input2_event, output_event });
        scatter_event.wait_and_throw();
        scatter_check(output_arr);
    }

private:
    std::int64_t count1_;
    std::int64_t count2_;
};

TEMPLATE_LIST_TEST_M(scatter_2d_test, "Scatter 2d", "[l2][norms][aux][small]", types_to_check) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_scatter();
}

template <typename Float>
class inner_product_test_random : public te::float_algo_fixture<Float> {
public:
    void generate() {
        init_val_ = GENERATE(0.1, 5.0);
        r_count1_ = GENERATE(17, 31);
        r_count2_ = GENERATE(7, 29);
        c_count_ = GENERATE(3, 13);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output() {
        return ndarray<Float, 2>::full(this->get_queue(), { r_count1_, r_count2_ }, init_val_);
    }

    void generate_input() {
        const auto input1_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ r_count1_, c_count_ }.fill_uniform(-0.2, 0.5));
        this->input_table1_ = input1_dataframe.get_table(this->get_homogen_table_id());
        const auto input2_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ r_count2_, c_count_ }.fill_uniform(-0.5, 1.0));
        this->input_table2_ = input2_dataframe.get_table(this->get_homogen_table_id());
    }

    void inner_product_check(const ndview<Float, 2>& out, const Float atol = 1.e-3) {
        for (std::int64_t i = 0; i < r_count1_; ++i) {
            const auto inp1_row =
                row_accessor<const Float>{ input_table1_ }.pull(this->get_queue(), { i, i + 1 });
            for (std::int64_t j = 0; j < r_count2_; ++j) {
                const auto inp2_row =
                    row_accessor<const Float>{ input_table2_ }.pull(this->get_queue(),
                                                                    { j, j + 1 });
                Float gtv = init_val_;
                for (std::int64_t k = 0; k < c_count_; ++k) {
                    gtv += Float(-2.0) * inp1_row[k] * inp2_row[k];
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

    void test_inner_product() {
        auto input1_arr = row_accessor<const Float>{ input_table1_ }.pull(this->get_queue());
        auto input2_arr = row_accessor<const Float>{ input_table2_ }.pull(this->get_queue());
        auto input1 = ndview<Float, 2>::wrap(input1_arr.get_data(), { r_count1_, c_count_ });
        auto input2 = ndview<Float, 2>::wrap(input2_arr.get_data(), { r_count2_, c_count_ });
        auto [output, output_event] = this->output();
        auto ip_event =
            compute_inner_product(this->get_queue(), input1, input2, output, { output_event });
        ip_event.wait_and_throw();
        inner_product_check(output);
    }

private:
    table input_table1_;
    table input_table2_;
    Float init_val_;
    std::int64_t c_count_;
    std::int64_t r_count1_;
    std::int64_t r_count2_;
};

TEMPLATE_LIST_TEST_M(inner_product_test_random,
                     "Randomly filled Inner-Product computation",
                     "[l2][ip][aux][small]",
                     types_to_check) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_inner_product();
}

} // namespace oneapi::dal::backend::primitives::test
