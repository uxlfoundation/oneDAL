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

#include "oneapi/dal/algo/linear_regression/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::linear_regression::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace linear_regression = oneapi::dal::linear_regression;

template <typename TestType>
class lr_online_spmd_test : public lr_test<TestType, lr_online_spmd_test<TestType>> {
public:
    using base_t = lr_test<TestType, lr_online_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::train_input_t;
    using partial_input_t = typename base_t::partial_input_t;
    using partial_result_t = typename base_t::partial_result_t;
    using result_t = typename base_t::train_result_t;

    void set_rank_count(std::int64_t rank_count) {
        n_rank = rank_count;
    }

    std::int64_t get_rank_count() {
        return n_rank;
    }

    void generate_dimensions() {
        this->t_count_ = GENERATE(307, 12999);
        this->s_count_ = GENERATE(10000);
        this->f_count_ = GENERATE(2, 17);
        this->r_count_ = GENERATE(2, 15);
        this->intercept_ = GENERATE(0, 1);
    }

    template <typename... Args>
    result_t finalize_train_override(Args&&... args) {
        return this->finalize_train_via_spmd_threads_and_merge(n_rank, std::forward<Args>(args)...);
    }

    result_t merge_finalize_train_result_override(const std::vector<result_t>& results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<partial_result_t> split_finalize_train_input_override(std::int64_t split_count,
                                                                      Args&&... args) {
        ONEDAL_ASSERT(split_count == n_rank);
        const std::vector<partial_result_t> input{ std::forward<Args>(args)... };

        return input;
    }

    void run_and_check_linear_online_spmd(std::int64_t n_rank,
                                          std::int64_t n_blocks,
                                          std::int64_t seed = 888,
                                          double tol = 1e-2) {
        table x_train, y_train, x_test, y_test;
        std::tie(x_train, y_train, x_test, y_test) = this->prepare_inputs(seed, tol);

        const auto desc = this->get_descriptor();
        std::vector<partial_result_t> partial_results;
        auto input_table_x = te::split_table_by_rows<float_t>(this->get_policy(), x_train, n_rank);
        auto input_table_y = te::split_table_by_rows<float_t>(this->get_policy(), y_train, n_rank);
        for (int64_t i = 0; i < n_rank; i++) {
            partial_result_t partial_result;
            auto input_table_x_blocks =
                te::split_table_by_rows<float_t>(this->get_policy(), input_table_x[i], n_blocks);
            auto input_table_y_blocks =
                te::split_table_by_rows<float_t>(this->get_policy(), input_table_y[i], n_blocks);
            for (int64_t j = 0; j < n_blocks; j++) {
                partial_result = this->partial_train(desc,
                                                     partial_result,
                                                     input_table_x_blocks[j],
                                                     input_table_y_blocks[j]);
            }
            partial_results.push_back(partial_result);
        }

        auto train_result = this->finalize_train_override(desc, partial_results);

        SECTION("Checking intercept values") {
            if (desc.get_result_options().test(result_options::intercept))
                base_t::check_if_close(train_result.get_intercept(), base_t::bias_, tol);
        }

        SECTION("Checking coefficient values") {
            if (desc.get_result_options().test(result_options::coefficients))
                base_t::check_if_close(train_result.get_coefficients(), base_t::beta_, tol);
        }

        train_result = this->finalize_train_override(desc, partial_results);

        SECTION("Checking intercept values after double finalize") {
            if (desc.get_result_options().test(result_options::intercept))
                base_t::check_if_close(train_result.get_intercept(), base_t::bias_, tol);
        }

        SECTION("Checking coefficient values after double finalize") {
            if (desc.get_result_options().test(result_options::coefficients))
                base_t::check_if_close(train_result.get_coefficients(), base_t::beta_, tol);
        }
    }

private:
    std::int64_t n_rank;
};

TEMPLATE_LIST_TEST_M(lr_online_spmd_test, "lr common flow", "[lr][integration][spmd]", lr_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate(777);

    this->set_rank_count(GENERATE(1, 2, 4));
    std::int64_t n_blocks = GENERATE(1, 3, 10);

    this->run_and_check_linear_online_spmd(this->get_rank_count(), n_blocks);
}

} // namespace oneapi::dal::linear_regression::test
