/*******************************************************************************
* Copyright 2023 Intel Corporation
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
class lr_train_params_test : public lr_test<TestType, lr_train_params_test<TestType>> {
public:
    using base_t = lr_test<TestType, lr_train_params_test<TestType>>;

    using task_t = typename base_t::task_t;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;

    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    void generate_dimensions() {
        this->t_count_ = GENERATE(711, 3072);
        this->s_count_ = GENERATE(611, 777);
        this->f_count_ = GENERATE(35, 45);
        this->r_count_ = GENERATE(17, 61);

        this->intercept_ = GENERATE(0, 1);
    }

    auto get_descriptor() const {
        return linear_regression::descriptor<float_t, method_t, task_t>(this->intercept_);
    }

    void generate_parameters() {
        /// Note: these are set in such a way that the upper end of 'max_cols_batched_'
        /// would not trigger for the given test sizes, while the upper end for
        /// 'small_rows_max_cols_batched_' would.
        this->block_ = GENERATE(512, 2048);
        this->max_cols_batched_ = GENERATE(50);
        this->small_rows_threshold_ = GENERATE(15, 70);
        this->small_rows_max_cols_batched_ = GENERATE(40);
        this->grain_size_ = GENERATE(1, 10);
        this->pack_as_struct_ = GENERATE(0, 1);
    }

    auto get_current_parameters() const {
        detail::train_parameters res{};
        res.set_cpu_macro_block(this->block_), res.set_gpu_macro_block(this->block_);
        res.set_cpu_max_cols_batched(this->max_cols_batched_);
        res.set_cpu_small_rows_threshold(this->small_rows_threshold_);
        res.set_cpu_small_rows_max_cols_batched(this->small_rows_max_cols_batched_);
        res.set_cpu_grain_size(this->grain_size_);
        return res;
    }

    template <typename Desc, typename... Args>
    train_result_t train_override(Desc&& desc, Args&&... args) {
        REQUIRE(this->block_ > 0);
        REQUIRE(this->max_cols_batched_ > 0);
        REQUIRE(this->small_rows_threshold_ > 0);
        REQUIRE(this->small_rows_max_cols_batched_ > 0);
        REQUIRE(this->grain_size_ > 0);
        const auto params = this->get_current_parameters();
        if (this->pack_as_struct_) {
            return te::float_algo_fixture<float_t>::train(
                std::forward<Desc>(desc),
                params,
                train_input_t{ std::forward<Args>(args)... });
        }
        else {
            return te::float_algo_fixture<float_t>::train(std::forward<Desc>(desc),
                                                          params,
                                                          std::forward<Args>(args)...);
        }
    }

private:
    std::int64_t block_;
    std::int64_t max_cols_batched_;
    std::int64_t small_rows_threshold_;
    std::int64_t small_rows_max_cols_batched_;
    std::int64_t grain_size_;
    bool pack_as_struct_;
};

TEMPLATE_LIST_TEST_M(lr_train_params_test, "LR train params", "[lr][train][params]", lr_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate(999);
    this->generate_parameters();

    this->run_and_check_linear();
}

} // namespace oneapi::dal::linear_regression::test
