/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train.hpp"
#include "oneapi/dal/algo/linear_regression/partial_train.hpp"
#include "oneapi/dal/algo/linear_regression/finalize_train.hpp"
#include "oneapi/dal/algo/linear_regression/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/test/engine/metrics/regression.hpp"

namespace oneapi::dal::linear_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType, typename Derived>
class lr_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;

    using train_input_t = train_input<task_t>;
    using train_result_t = train_result<task_t>;
    using test_input_t = infer_input<task_t>;
    using test_result_t = infer_result<task_t>;

    using partial_input_t = partial_train_input<>;
    using partial_result_t = partial_train_result<>;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    Derived* get_impl() {
        return static_cast<Derived*>(this);
    }

    table compute_responses(const table& beta, const table& bias, const table& data) const {
        const auto s_count = data.get_row_count();

        REQUIRE(beta.get_row_count() == this->r_count_);
        REQUIRE(beta.get_column_count() == this->f_count_);

        REQUIRE(bias.get_row_count() == std::int64_t(1));
        REQUIRE(bias.get_column_count() == this->r_count_);

        auto res_arr = array<float_t>::zeros(s_count * this->r_count_);

        const auto beta_arr = row_accessor<const float_t>(beta).pull({ 0, -1 });
        const auto bias_arr = row_accessor<const float_t>(bias).pull({ 0, -1 });
        const auto data_arr = row_accessor<const float_t>(data).pull({ 0, -1 });

        for (std::int64_t s = 0; s < s_count; ++s) {
            for (std::int64_t r = 0; r < this->r_count_; ++r) {
                for (std::int64_t f = 0; f < this->f_count_; ++f) {
                    const auto& v = data_arr[s * this->f_count_ + f];
                    const auto& b = beta_arr[r * this->f_count_ + f];
                    *(res_arr.get_mutable_data() + s * this->r_count_ + r) += v * b;
                }
            }
        }

        for (std::int64_t s = 0; s < s_count; ++s) {
            for (std::int64_t r = 0; r < this->r_count_; ++r) {
                *(res_arr.get_mutable_data() + s * this->r_count_ + r) += bias_arr[r];
            }
        }

        return homogen_table::wrap(res_arr, s_count, this->r_count_);
    }

    std::tuple<table, table> generate_betas(std::int64_t seed) const {
        std::mt19937 meta_gen(seed);
        std::tuple<table, table> result{ {}, {} };

        const std::int64_t betas_seed = meta_gen();
        const auto betas_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->r_count_, this->f_count_ }.fill_uniform(-10.1,
                                                                                 10.1,
                                                                                 betas_seed));
        std::get<0>(result) = betas_dataframe.get_table(this->get_homogen_table_id());

        if (this->intercept_) {
            const std::int64_t bias_seed = meta_gen();
            const auto bias_dataframe = GENERATE_DATAFRAME(
                te::dataframe_builder{ std::int64_t(1), this->r_count_ }.fill_uniform(-15.5,
                                                                                      15.5,
                                                                                      bias_seed));
            std::get<1>(result) = bias_dataframe.get_table(this->get_homogen_table_id());
        }
        else {
            auto bias_arr = array<float_t>::zeros(this->r_count_);
            std::get<1>(result) = homogen_table::wrap(bias_arr, std::int64_t(1), this->r_count_);
        }

        return result;
    }

    double generate_alpha(std::int64_t seed) const {
        std::mt19937 gen(seed);

        double alpha_min = 1;
        double alpha_max = 5;

        std::uniform_real_distribution<double> dist(alpha_min, alpha_max);

        return dist(gen);
    }

    void check_table_dimensions(const table& x_train,
                                const table& y_train,
                                const table& x_test,
                                const table& y_test) {
        REQUIRE(x_train.get_column_count() == this->f_count_);
        REQUIRE(x_train.get_row_count() == this->s_count_);
        REQUIRE(x_test.get_column_count() == this->f_count_);
        REQUIRE(x_test.get_row_count() == this->t_count_);
        REQUIRE(y_train.get_column_count() == this->r_count_);
        REQUIRE(y_train.get_row_count() == this->s_count_);
        REQUIRE(y_test.get_column_count() == this->r_count_);
        REQUIRE(y_test.get_row_count() == this->t_count_);
    }

    void generate(std::int64_t seed = 777) {
        this->get_impl()->generate_dimensions();

        auto [beta, bias] = generate_betas(seed);

        this->bias_ = std::move(bias);
        this->beta_ = std::move(beta);
        this->alpha_ = generate_alpha(seed);
    }

    auto get_descriptor(double alpha = 0.0) const {
        result_option_id resopts = result_options::coefficients;
        if (this->intercept_)
            resopts = resopts | result_options::intercept;
        return linear_regression::descriptor<float_t, method_t, task_t>(intercept_, alpha)
            .set_result_options(resopts);
    }

    void check_if_close(const table& left, const table& right, double tol = 1e-3) {
        constexpr auto eps = std::numeric_limits<float_t>::epsilon();

        const auto c_count = left.get_column_count();
        const auto r_count = left.get_row_count();

        REQUIRE(right.get_column_count() == c_count);
        REQUIRE(right.get_row_count() == r_count);

        row_accessor<const float_t> lacc(left);
        row_accessor<const float_t> racc(right);

        const auto larr = lacc.pull({ 0, -1 });
        const auto rarr = racc.pull({ 0, -1 });

        for (std::int64_t r = 0; r < r_count; ++r) {
            for (std::int64_t c = 0; c < c_count; ++c) {
                const auto lval = larr[r * c_count + c];
                const auto rval = rarr[r * c_count + c];

                CAPTURE(r_count, c_count, r, c, lval, rval);

                const auto aerr = std::abs(lval - rval);
                if (aerr < tol)
                    continue;

                const auto den = std::max({ eps, //
                                            std::abs(lval),
                                            std::abs(rval) });

                const auto rerr = aerr / den;
                CAPTURE(aerr, rerr, den, r, c, lval, rval);
                REQUIRE(rerr < tol);
            }
        }
    }

    void check_coefficient_shrinkage(const table& lr_coeffs,
                                     const table& rr_coeffs,
                                     double tol = 1e-3) {
        row_accessor<const float_t> lr_acc(lr_coeffs);
        row_accessor<const float_t> rr_acc(rr_coeffs);
        const auto lr_arr = lr_acc.pull({ 0, -1 });
        const auto rr_arr = rr_acc.pull({ 0, -1 });

        double lr_norm_squared = 0, rr_norm_squared = 0;
        for (std::int64_t i = 0; i < lr_arr.get_count(); ++i) {
            lr_norm_squared += lr_arr[i] * lr_arr[i];
            rr_norm_squared += rr_arr[i] * rr_arr[i];
        }

        REQUIRE(rr_norm_squared <= lr_norm_squared + tol);
    }

    std::tuple<table, table, table, table> prepare_inputs(std::int64_t seed = 888,
                                                          double tol = 1e-2) {
        using namespace ::oneapi::dal::detail;

        std::mt19937 meta_gen(seed);

        const std::int64_t train_seed = meta_gen();
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->s_count_, this->f_count_ }.fill_uniform(-5.5,
                                                                                 3.5,
                                                                                 train_seed));
        auto x_train = train_dataframe.get_table(this->get_homogen_table_id());

        const std::int64_t test_seed = meta_gen();
        const auto test_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->t_count_, this->f_count_ }.fill_uniform(-3.5,
                                                                                 5.5,
                                                                                 test_seed));
        auto x_test = test_dataframe.get_table(this->get_homogen_table_id());

        auto y_train = compute_responses(this->beta_, this->bias_, x_train);
        auto y_test = compute_responses(this->beta_, this->bias_, x_test);

        check_table_dimensions(x_train, y_train, x_test, y_test);
        return { x_train, y_train, x_test, y_test };
    }

    void run_and_check_ridge(std::int64_t seed = 888, double tol = 1e-2) {
        table x_train, y_train, x_test, y_test;
        std::tie(x_train, y_train, x_test, y_test) = prepare_inputs(seed, tol);

        const auto linear_desc = this->get_descriptor();
        const auto linear_train_res = this->train(linear_desc, x_train, y_train);

        const auto ridge_desc = this->get_descriptor(this->alpha_);
        const auto ridge_train_res = this->train(ridge_desc, x_train, y_train);

        SECTION("Checking coefficient shrinkage") {
            this->check_coefficient_shrinkage(linear_train_res.get_coefficients(),
                                              ridge_train_res.get_coefficients(),
                                              tol);
        }
    }

    void run_and_check_linear(std::int64_t seed = 888, double tol = 1e-2) {
        table x_train, y_train, x_test, y_test;
        std::tie(x_train, y_train, x_test, y_test) = prepare_inputs(seed, tol);

        const auto desc = this->get_descriptor();
        const auto train_res = this->train(desc, x_train, y_train);

        SECTION("Checking intercept values") {
            if (desc.get_result_options().test(result_options::intercept))
                check_if_close(train_res.get_intercept(), this->bias_, tol);
        }

        SECTION("Checking coefficient values") {
            if (desc.get_result_options().test(result_options::coefficients))
                check_if_close(train_res.get_coefficients(), this->beta_, tol);
        }

        const auto infer_res = this->infer(desc, x_test, train_res.get_model());

        SECTION("Checking infer results") {
            check_if_close(infer_res.get_responses(), y_test, tol);
        }
    }

    /* Note: difference between this test and the above, is that the linear system to solve
    here is not positive-definite, thus it has an infinite number of possible solutions. The
    solution here is the one with minimum norm, which is typically more desirable. */
    void run_and_check_linear_indefinite(double tol = 1e-3) {
        const auto dtype =
            std::is_same<float_t, double>::value ? data_type::float64 : data_type::float32;
        const float_t X[] = { -0.98912135, -0.36778665, 1.28792526, 0.19397442,  0.9202309,
                              0.57710379,  -0.63646365, 0.54195222, -0.31659545, -0.32238912,
                              0.09716732,  -1.52593041, 1.1921661,  -0.67108968, 1.00026942 };
        const float_t y[] = { 0.13632112, 1.53203308, -0.65996941 };
        auto X_tbl = oneapi::dal::detail::homogen_table_builder()
                         .set_data_type(dtype)
                         .set_layout(data_layout::row_major)
                         .allocate(3, 5)
                         .copy_data(X, 3, 5)
                         .build();
        auto y_tbl = oneapi::dal::detail::homogen_table_builder()
                         .set_data_type(dtype)
                         .set_layout(data_layout::row_major)
                         .allocate(3, 1)
                         .copy_data(y, 3, 1)
                         .build();

        auto desc = this->get_descriptor();
        auto train_res = this->train(desc, X_tbl, y_tbl);
        const auto coefs = train_res.get_coefficients();

        if (desc.get_result_options().test(result_options::intercept)) {
            const float_t expected_beta[] = { 0.27785494,
                                              0.53011669,
                                              0.34352259,
                                              0.40506216,
                                              -1.26026447 };
            const float_t expected_intercept[] = { 1.24485441 };
            const auto expected_beta_tbl = oneapi::dal::detail::homogen_table_builder()
                                               .set_data_type(dtype)
                                               .set_layout(data_layout::row_major)
                                               .allocate(1, 5)
                                               .copy_data(expected_beta, 1, 5)
                                               .build();
            const auto expected_intercept_tbl = oneapi::dal::detail::homogen_table_builder()
                                                    .set_data_type(dtype)
                                                    .set_layout(data_layout::row_major)
                                                    .allocate(1, 1)
                                                    .copy_data(expected_intercept, 1, 1)
                                                    .build();

            const auto intercept = train_res.get_intercept();

            SECTION("Checking intercept values") {
                check_if_close(intercept, expected_intercept_tbl, tol);
            }
            SECTION("Checking coefficient values") {
                check_if_close(coefs, expected_beta_tbl, tol);
            }
        }

        else {
            const float_t expected_beta[] = { 0.38217445,
                                              0.2732197,
                                              1.87135517,
                                              0.63458468,
                                              -2.08473134 };
            const auto expected_beta_tbl = oneapi::dal::detail::homogen_table_builder()
                                               .set_data_type(dtype)
                                               .set_layout(data_layout::row_major)
                                               .allocate(1, 5)
                                               .copy_data(expected_beta, 1, 5)
                                               .build();
            SECTION("Checking coefficient values") {
                check_if_close(coefs, expected_beta_tbl, tol);
            }
        }
    }

    void run_and_check_linear_indefinite_multioutput(double tol = 1e-3) {
        const auto dtype =
            std::is_same<float_t, double>::value ? data_type::float64 : data_type::float32;
        const float_t X[] = { -0.98912135, -0.36778665, 1.28792526, 0.19397442,  0.9202309,
                              0.57710379,  -0.63646365, 0.54195222, -0.31659545, -0.32238912,
                              0.09716732,  -1.52593041, 1.1921661,  -0.67108968, 1.00026942 };
        const float_t y[] = { 0.13632112,  1.53203308, -0.65996941,
                              -0.31179486, 0.33776913, -2.2074711 };
        auto X_tbl = oneapi::dal::detail::homogen_table_builder()
                         .set_data_type(dtype)
                         .set_layout(data_layout::row_major)
                         .allocate(3, 5)
                         .copy_data(X, 3, 5)
                         .build();
        auto y_tbl = oneapi::dal::detail::homogen_table_builder()
                         .set_data_type(dtype)
                         .set_layout(data_layout::row_major)
                         .allocate(3, 2)
                         .copy_data(y, 3, 2)
                         .build();

        auto desc = this->get_descriptor();
        auto train_res = this->train(desc, X_tbl, y_tbl);
        const auto coefs = train_res.get_coefficients();

        if (desc.get_result_options().test(result_options::intercept)) {
            const float_t expected_beta[] = {
                -0.18692112, -0.20034801, -0.09590892, -0.13672683, 0.56229012,
                -0.97006008, 1.39413595,  0.49238012,  1.11041239,  -0.79213452,
            };
            const float_t expected_intercept[] = { -0.48964358, 0.96467681 };
            const auto expected_beta_tbl = oneapi::dal::detail::homogen_table_builder()
                                               .set_data_type(dtype)
                                               .set_layout(data_layout::row_major)
                                               .allocate(2, 5)
                                               .copy_data(expected_beta, 2, 5)
                                               .build();
            const auto expected_intercept_tbl = oneapi::dal::detail::homogen_table_builder()
                                                    .set_data_type(dtype)
                                                    .set_layout(data_layout::row_major)
                                                    .allocate(1, 2)
                                                    .copy_data(expected_intercept, 1, 2)
                                                    .build();

            const auto intercept = train_res.get_intercept();

            SECTION("Checking intercept values") {
                check_if_close(intercept, expected_intercept_tbl, tol);
            }
            SECTION("Checking coefficient values") {
                check_if_close(coefs, expected_beta_tbl, tol);
            }
        }

        else {
            const float_t expected_beta[] = { -0.22795353, -0.09930168, -0.69685744, -0.22700585,
                                              0.88658098,  -0.88921961, 1.19505839,  1.67634561,
                                              1.2882766,   -1.43103981 };
            const auto expected_beta_tbl = oneapi::dal::detail::homogen_table_builder()
                                               .set_data_type(dtype)
                                               .set_layout(data_layout::row_major)
                                               .allocate(2, 5)
                                               .copy_data(expected_beta, 2, 5)
                                               .build();
            SECTION("Checking coefficient values") {
                check_if_close(coefs, expected_beta_tbl, tol);
            }
        }
    }

    template <typename Float>
    std::vector<dal::table> split_table_by_rows(const dal::table& t, std::int64_t split_count) {
        ONEDAL_ASSERT(0l < split_count);
        ONEDAL_ASSERT(split_count <= t.get_row_count());

        const std::int64_t row_count = t.get_row_count();
        const std::int64_t column_count = t.get_column_count();
        const std::int64_t block_size_regular = row_count / split_count;
        const std::int64_t block_size_tail = row_count % split_count;

        std::vector<dal::table> result(split_count);

        std::int64_t row_offset = 0;
        for (std::int64_t i = 0; i < split_count; i++) {
            const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
            const std::int64_t block_size = block_size_regular + tail;

            const auto row_range = dal::range{ row_offset, row_offset + block_size };
            const auto block = dal::row_accessor<const Float>{ t }.pull(row_range);
            result[i] = dal::homogen_table::wrap(block, block_size, column_count);
            row_offset += block_size;
        }

        return result;
    }

    void run_and_check_linear_online(std::int64_t nBlocks) {
        std::int64_t seed = 888;
        double tol = 1e-2;
        table x_train, y_train, x_test, y_test;
        std::tie(x_train, y_train, x_test, y_test) = prepare_inputs(seed, tol);

        const auto desc = this->get_descriptor();
        dal::linear_regression::partial_train_result<> partial_result;
        auto input_table_x = split_table_by_rows<double>(x_train, nBlocks);
        auto input_table_y = split_table_by_rows<double>(y_train, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; i++) {
            partial_result =
                this->partial_train(desc, partial_result, input_table_x[i], input_table_y[i]);
        }
        auto train_res = this->finalize_train(desc, partial_result);

        SECTION("Checking intercept values") {
            if (desc.get_result_options().test(result_options::intercept))
                check_if_close(train_res.get_intercept(), this->bias_, tol);
        }

        SECTION("Checking coefficient values") {
            if (desc.get_result_options().test(result_options::coefficients))
                check_if_close(train_res.get_coefficients(), this->beta_, tol);
        }

        const auto infer_res = this->infer(desc, x_test, train_res.get_model());

        SECTION("Checking infer results") {
            check_if_close(infer_res.get_responses(), y_test, tol);
        }
    }

    void run_and_check_ridge_online(std::int64_t nBlocks) {
        std::int64_t seed = 888;
        double tol = 1e-2;
        table x_train, y_train, x_test, y_test;
        std::tie(x_train, y_train, x_test, y_test) = prepare_inputs(seed, tol);

        auto input_table_x = split_table_by_rows<double>(x_train, nBlocks);
        auto input_table_y = split_table_by_rows<double>(y_train, nBlocks);

        const auto linear_desc = this->get_descriptor();
        dal::linear_regression::partial_train_result<> linear_partial_result;
        for (std::int64_t i = 0; i < nBlocks; i++) {
            linear_partial_result = this->partial_train(linear_desc,
                                                        linear_partial_result,
                                                        input_table_x[i],
                                                        input_table_y[i]);
        }
        auto linear_train_res = this->finalize_train(linear_desc, linear_partial_result);

        const auto ridge_desc = this->get_descriptor(this->alpha_);
        dal::linear_regression::partial_train_result<> ridge_partial_result;
        for (std::int64_t i = 0; i < nBlocks; i++) {
            ridge_partial_result = this->partial_train(ridge_desc,
                                                       ridge_partial_result,
                                                       input_table_x[i],
                                                       input_table_y[i]);
        }
        auto ridge_train_res = this->finalize_train(ridge_desc, ridge_partial_result);

        SECTION("Checking coefficient shrinkage") {
            this->check_coefficient_shrinkage(linear_train_res.get_coefficients(),
                                              ridge_train_res.get_coefficients(),
                                              tol);
        }
    }

protected:
    bool intercept_ = true;
    float_t alpha_;
    std::int64_t t_count_;
    std::int64_t s_count_;
    std::int64_t f_count_;
    std::int64_t r_count_;

    table bias_;
    table beta_;
};

using lr_types = COMBINE_TYPES((float, double),
                               (linear_regression::method::norm_eq),
                               (linear_regression::task::regression));

} // namespace oneapi::dal::linear_regression::test
