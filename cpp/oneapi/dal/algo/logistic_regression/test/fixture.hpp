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

#include <cmath>
#include <random>

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train.hpp"
#include "oneapi/dal/algo/logistic_regression/infer.hpp"
#include "oneapi/dal/algo/newton_cg/common.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/csr_table_builder.hpp"

namespace oneapi::dal::logistic_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType, typename Derived>
class log_reg_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;

    using train_input_t = train_input<task_t>;
    using train_result_t = train_result<task_t>;
    using test_input_t = infer_input<task_t>;
    using test_result_t = infer_result<task_t>;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    Derived* get_impl() {
        return static_cast<Derived*>(this);
    }

    bool is_sparse() {
        return std::is_same_v<method_t, logistic_regression::method::sparse>;
    }

    auto get_descriptor(double tol = 1e-4, std::int64_t maxiter = 100) const {
        result_option_id resopts = result_options::coefficients;

        auto optimizer_desc = dal::newton_cg::descriptor<float_t>(tol, maxiter);

        if (this->fit_intercept_)
            resopts = resopts | result_options::intercept;
        return logistic_regression::descriptor<float_t, method_t, task_t>(fit_intercept_, C_)
            .set_result_options(resopts)
            .set_optimizer(optimizer_desc);
    }

    virtual void gen_dimensions(std::int64_t n = -1,
                                std::int64_t p = -1,
                                double train_size_coef = 0.7) {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(100, 200, 1000, 10000, 50000);
            this->p_ = GENERATE(10, 20, 30);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }
        this->train_size_ = (this->n_ * train_size_coef);
        this->test_size_ = this->n_ - this->train_size_;
    }

    float_t predict_proba(float_t* ptr, float_t* params_ptr, float_t intercept) {
        float_t val = 0;
        for (std::int64_t j = 0; j < p_; ++j) {
            val += ptr[j] * params_ptr[j];
        }
        val += intercept;
        return float_t(1.0) / (1.0 + std::exp(-val));
    }

    void gen_input(bool fit_intercept = true, double C = 1.0, std::int64_t seed = 2007) {
        this->fit_intercept_ = fit_intercept;
        this->C_ = C;

        std::int64_t dim = fit_intercept_ ? p_ + 1 : p_;

        std::mt19937 rnd(seed + n_ + p_);

        std::uniform_real_distribution<> dis_params(-3.0, 3.0);

        if (this->is_sparse()) {
            auto builder_train =
                te::csr_table_builder<float_t>(train_size_, p_, 0.5, sparse_indexing::zero_based);
            this->X_train_ = builder_train.build_csr_table(this->get_policy());

            auto builder_test =
                te::csr_table_builder<float_t>(test_size_, p_, 0.5, sparse_indexing::zero_based);
            this->X_test_ = builder_test.build_csr_table(this->get_policy());

            table dense_train = builder_train.build_dense_table(this->get_policy());
            table dense_test = builder_test.build_dense_table(this->get_policy());

            X_train_host_ = row_accessor<const float_t>(dense_train)
                                .pull(this->get_queue(), { 0, -1 }, sycl::usm::alloc::host);
            X_test_host_ = row_accessor<const float_t>(dense_test)
                               .pull(this->get_queue(), { 0, -1 }, sycl::usm::alloc::host);
        }
        else {
            std::uniform_real_distribution<> dis_data(-10.0, 10.0);
            X_train_host_ = array<float_t>::zeros(train_size_ * p_);
            X_test_host_ = array<float_t>::zeros(test_size_ * p_);

            auto* x_ptr = X_train_host_.get_mutable_data();
            for (std::int64_t i = 0; i < train_size_; ++i) {
                for (std::int64_t j = 0; j < p_; ++j) {
                    *(x_ptr + i * p_ + j) = dis_data(rnd);
                }
            }
            x_ptr = X_test_host_.get_mutable_data();
            for (std::int64_t i = 0; i < test_size_; ++i) {
                for (std::int64_t j = 0; j < p_; ++j) {
                    *(x_ptr + i * p_ + j) = dis_data(rnd);
                }
            }
            X_train_ =
                homogen_table::wrap<float_t>(X_train_host_.get_mutable_data(), train_size_, p_);
            X_test_ = homogen_table::wrap<float_t>(X_test_host_.get_mutable_data(), test_size_, p_);
        }

        auto* x_ptr = X_train_host_.get_mutable_data();
        y_train_host_ = array<std::int32_t>::zeros(train_size_);
        auto* y_ptr = y_train_host_.get_mutable_data();

        params_host_ = array<float_t>::zeros(dim);
        auto* params_ptr = params_host_.get_mutable_data();

        for (std::int64_t i = 0; i < dim; ++i) {
            *(params_ptr + i) = dis_params(rnd);
        }

        constexpr float_t half = 0.5;
        for (std::int64_t i = 0; i < train_size_; ++i) {
            float_t val = predict_proba(x_ptr + i * p_,
                                        params_ptr + (std::int64_t)fit_intercept_,
                                        fit_intercept_ ? *params_ptr : 0);
            y_ptr[i] = bool(val < half);
        }

        x_ptr = X_test_host_.get_mutable_data();
        y_test_host_ = array<std::int32_t>::zeros(test_size_);
        y_ptr = y_test_host_.get_mutable_data();

        for (std::int64_t i = 0; i < test_size_; ++i) {
            float_t val = predict_proba(x_ptr + i * p_,
                                        params_ptr + (std::int64_t)fit_intercept_,
                                        fit_intercept_ ? *params_ptr : 0);
            y_ptr[i] = bool(val < half);
        }
    }

    void run_test(double tol = 1e-4, std::int64_t maxiter = 100) {
        table y_train =
            homogen_table::wrap<std::int32_t>(y_train_host_.get_mutable_data(), train_size_, 1);
        table y_table =
            homogen_table::wrap<std::int32_t>(y_test_host_.get_mutable_data(), test_size_, 1);
        const auto desc = this->get_descriptor(tol, maxiter);
        const auto train_res = this->train(desc, X_train_, y_train);
        table intercept;
        array<float_t> bias_host;
        if (fit_intercept_) {
            intercept = train_res.get_intercept();
            bias_host = row_accessor<const float_t>(intercept).pull({ 0, -1 });
        }
        table coefs = train_res.get_coefficients();
        auto coefs_host = row_accessor<const float_t>(coefs).pull({ 0, -1 });

        std::int64_t train_acc = 0;
        std::int64_t test_acc = 0;

        const auto infer_res = this->infer(desc, X_test_, train_res.get_model());

        table resp_table = infer_res.get_responses();
        auto resp_host = row_accessor<const std::int32_t>(resp_table).pull({ 0, -1 });

        table prob_table = infer_res.get_probabilities();
        auto prob_host = row_accessor<const float_t>(prob_table).pull({ 0, -1 });

        for (std::int64_t i = 0; i < train_size_; ++i) {
            float_t val = predict_proba(X_train_host_.get_mutable_data() + i * p_,
                                        coefs_host.get_mutable_data(),
                                        fit_intercept_ ? *bias_host.get_mutable_data() : 0);
            std::int32_t resp = val >= 0.5 ? 1 : 0;
            if (resp == *(y_train_host_.get_mutable_data() + i)) {
                train_acc += 1;
            }
        }

        std::int64_t acc_algo = 0;
        for (std::int64_t i = 0; i < test_size_; ++i) {
            float_t val = predict_proba(X_test_host_.get_mutable_data() + i * p_,
                                        coefs_host.get_mutable_data(),
                                        fit_intercept_ ? *bias_host.get_mutable_data() : 0);
            std::int32_t resp = val >= 0.5 ? 1 : 0;
            if (resp == *(y_test_host_.get_mutable_data() + i)) {
                test_acc += 1;
            }

            REQUIRE(abs(val - *(prob_host.get_mutable_data() + i)) < 1e-5);
            REQUIRE(*(resp_host.get_mutable_data() + i) == resp);

            if (*(resp_host.get_mutable_data() + i) == *(y_test_host_.get_mutable_data() + i)) {
                acc_algo++;
            }
        }

        float_t min_train_acc = 0.95;
        float_t min_test_acc = n_ < 500 ? 0.7 : 0.85;
        if (this->is_sparse() && n_ < 500) {
            min_test_acc = 0.65;
        }

        REQUIRE(train_size_ * min_train_acc < train_acc);
        REQUIRE(test_size_ * min_test_acc < test_acc);
        REQUIRE(test_size_ * min_test_acc < acc_algo);
        REQUIRE(test_acc == acc_algo);
    }

protected:
    bool fit_intercept_ = true;
    double C_ = 1.0;
    std::int64_t n_ = 0;
    std::int64_t p_ = 0;
    std::int64_t train_size_ = 0;
    std::int64_t test_size_ = 0;
    array<float_t> X_train_host_;
    array<float_t> X_test_host_;
    array<float_t> params_host_;
    array<std::int32_t> y_train_host_;
    array<std::int32_t> y_test_host_;

    array<std::int32_t> resp_train_;
    array<std::int32_t> resp_test_;
    table X_train_;
    table X_test_;
};

using log_reg_types = COMBINE_TYPES((float, double),
                                    (logistic_regression::method::dense_batch,
                                     logistic_regression::method::sparse),
                                    (logistic_regression::task::classification));

} // namespace oneapi::dal::logistic_regression::test
