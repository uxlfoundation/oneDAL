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

#include <type_traits>

#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class gemv_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr ndorder ao = std::tuple_element_t<1, Param>::value;

    gemv_test() {
        m_ = 0;
        n_ = 0;
    }

    void generate_small_dimensions() {
        m_ = GENERATE(3, 4, 5);
        n_ = GENERATE(4, 5, 6);
        CAPTURE(m_, n_);
    }

    void generate_medium_dimensions() {
        m_ = GENERATE(300, 400);
        n_ = GENERATE(400, 500);
        CAPTURE(m_, n_);
    }

    auto A() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { m_, n_ });
    }

    auto At() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { n_, m_ });
    }

    auto X() {
        check_if_initialized();
        return ndarray<float_t, 1>::ones(this->get_queue(), { n_ });
    }

    auto Y() {
        check_if_initialized();
        return ndarray<float_t, 1>::empty(this->get_queue(), { m_ });
    }

    auto Y_ones() {
        check_if_initialized();
        return ndarray<float_t, 1>::ones(this->get_queue(), { m_ });
    }

    void test_gemv() {
        auto y = Y();
        auto [y_ones, y_ones_e] = Y_ones();
        auto [a, a_e] = A();
        auto [x, x_e] = X();
        auto [at, at_e] = At();

        SECTION("Ax") {
            gemv(this->get_queue(), a, x, y, { a_e, x_e }).wait_and_throw();
            check_value_vector(y, m_, n_);
        }

        SECTION("At x") {
            gemv(this->get_queue(), at.t(), x, y, { at_e, x_e }).wait_and_throw();
            check_value_vector(y, m_, n_);
        }

        SECTION("Ax alpha + beta") {
            const float_t alpha = 2;
            const float_t beta = 3;

            gemv(this->get_queue(), a, x, y_ones, alpha, beta, { a_e, x_e, y_ones_e })
                .wait_and_throw();
            check_value_vector(y_ones, m_, n_ * alpha + beta);
        }

        SECTION("At x alpha + beta") {
            const float_t alpha = 2;
            const float_t beta = 3;

            gemv(this->get_queue(), at.t(), x, y_ones, alpha, beta, { a_e, x_e, y_ones_e })
                .wait_and_throw();
            check_value_vector(y_ones, m_, n_ * alpha + beta);
        }
    }

    void check_value_vector(const ndarray<float_t, 1>& vec, int64_t len, float_t value) {
        check_if_initialized();
        CAPTURE(vec.get_shape()[0]);
        REQUIRE(vec.get_shape() == ndshape<1>{ len });
        const float_t* ptr = vec.get_data();
        for (std::int64_t i = 0; i < vec.get_count(); i++) {
            if (std::int64_t(ptr[i]) != value) {
                CAPTURE(i, ptr[i]);
                FAIL();
            }
        }
        SUCCEED();
    }

    bool is_initialized() const {
        return m_ > 0 && n_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "gemv test is not initialized" };
        }
    }

private:
    std::int64_t m_;
    std::int64_t n_;
};

using gemv_types = COMBINE_TYPES((float, double), (c_order, f_order));

TEMPLATE_LIST_TEST_M(gemv_test, "ones matrix gemv on small sizes", "[gemv][small]", gemv_types) {
    // DPC++ GEMV from micro MKL libs is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_gemv();
}

TEMPLATE_LIST_TEST_M(gemv_test, "ones matrix gemv on medium sizes", "[gemv][small]", gemv_types) {
    // DPC++ GEMV from micro MKL libs is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    this->generate_medium_dimensions();
    this->test_gemv();
}

} // namespace oneapi::dal::backend::primitives::test
