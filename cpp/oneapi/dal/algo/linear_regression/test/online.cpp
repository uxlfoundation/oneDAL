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

#include "oneapi/dal/algo/linear_regression/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::linear_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class lr_online_test : public lr_test<TestType, lr_online_test<TestType>> {
public:
    using base_t = lr_test<TestType, lr_online_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    void generate_dimensions() {
        this->t_count_ = GENERATE(307, 1500, 2500);
        this->s_count_ = GENERATE(111, 444);
        this->f_count_ = GENERATE(2, 17);
        this->r_count_ = GENERATE(2, 15);
        this->intercept_ = GENERATE(0, 1);
        if (this->get_policy().is_cpu()) {
            this->use_non_batched_route = GENERATE(false, true);
        }
    }
};

TEMPLATE_LIST_TEST_M(lr_online_test, "LR common flow", "[lr][online]", lr_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate(777);
    const int64_t nBlocks = GENERATE(1, 3, 5, 8);

    this->run_and_check_linear_online(nBlocks);
}

TEMPLATE_LIST_TEST_M(lr_online_test, "RR common flow", "[rr][online]", lr_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate(777);
    const int64_t nBlocks = GENERATE(1, 3, 5, 8);

    this->run_and_check_ridge_online(nBlocks);
}

// finalize_train must leave the partial result as it was, so it can be finalized again
// or extended with more blocks; ridge used to add alpha to the partial xtx on GPU.
TEMPLATE_LIST_TEST_M(lr_online_test,
                     "LR and RR finalize twice on the same partial result",
                     "[lr][rr][online]",
                     lr_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    this->intercept_ = GENERATE(false, true);
    const double alpha = GENERATE(0.0, 1.5);
    const int64_t nBlocks = GENERATE(1, 3);

    const te::dataframe x_data = te::dataframe_builder{ 6, 2 }.fill_uniform(-5, 5, 7777).build();
    const te::dataframe y_data = te::dataframe_builder{ 6, 1 }.fill_uniform(-5, 5, 8888).build();
    const auto x = x_data.get_table(this->get_policy(), this->get_homogen_table_id());
    const auto y = y_data.get_table(this->get_policy(), this->get_homogen_table_id());
    const auto x_blocks = te::split_table_by_rows<float_t>(this->get_policy(), x, nBlocks);
    const auto y_blocks = te::split_table_by_rows<float_t>(this->get_policy(), y, nBlocks);

    const auto desc = this->get_descriptor(alpha);
    dal::linear_regression::partial_train_result<> partial;
    for (std::int64_t i = 0; i < nBlocks; i++) {
        partial = this->partial_train(desc, partial, x_blocks[i], y_blocks[i]);
    }
    const auto xtx = te::copy_table_values<float_t>(partial.get_partial_xtx());
    const auto xty = te::copy_table_values<float_t>(partial.get_partial_xty());

    const auto first = this->finalize_train(desc, partial);
    REQUIRE(te::copy_table_values<float_t>(partial.get_partial_xtx()) == xtx);
    REQUIRE(te::copy_table_values<float_t>(partial.get_partial_xty()) == xty);

    const auto second = this->finalize_train(desc, partial);
    te::check_if_table_content_equal<float_t>(second.get_coefficients(), first.get_coefficients());
    if (this->intercept_) {
        te::check_if_table_content_equal<float_t>(second.get_intercept(), first.get_intercept());
    }
}

} // namespace oneapi::dal::linear_regression::test
