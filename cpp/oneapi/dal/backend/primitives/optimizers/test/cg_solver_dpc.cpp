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

#include <type_traits>
#include "oneapi/dal/backend/primitives/optimizers/test/fixture.hpp"
#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include <math.h>

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Param>
class cg_solver_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void gen_input(std::int64_t n = -1) {
        if (n == -1) {
            n_ = GENERATE(5, 14, 25, 50, 100);
        }
        else {
            n_ = n;
        }
        A_host_ = ndarray<float_t, 2>::empty(this->get_queue(), { n_, n_ }, sycl::usm::alloc::host);
        x_host_ = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        b_host_ = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);

        primitives::host_engine eng(4014 + n_);
        primitives::uniform<float_t>(n_, x_host_.get_mutable_data(), eng, -1.0, 1.0);

        create_stable_matrix(this->get_queue(), A_host_);

        for (std::int64_t i = 0; i < n_; ++i) {
            b_host_.at(i) = 0;
            for (std::int64_t j = 0; j < n_; ++j) {
                b_host_.at(i) += A_host_.at(i, j) * x_host_.at(j);
            }
        }
    }

    void test_cg_solver() {
        auto A = A_host_.to_device(this->get_queue());
        auto b = b_host_.to_device(this->get_queue());

        linear_matrix_operator<float_t> mul_operator(this->get_queue(), A);
        auto [x0, x0_init_event] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { n_ }, sycl::usm::alloc::device);
        x0_init_event.wait_and_throw();
        auto buffer =
            ndarray<float_t, 1>::empty(this->get_queue(), { 3 * n_ }, sycl::usm::alloc::device);
        auto buffer1 = buffer.get_slice(0, n_);
        auto buffer2 = buffer.get_slice(n_, 2 * n_);
        auto buffer3 = buffer.get_slice(2 * n_, 3 * n_);

        auto [cg_event, inner_iter] = cg_solve(this->get_queue(),
                                               mul_operator,
                                               b,
                                               x0,
                                               buffer1,
                                               buffer2,
                                               buffer3,
                                               float_t(1e-6),
                                               float_t(1e-5),
                                               n_,
                                               {});
        cg_event.wait_and_throw();
        auto answer_host = x0.to_host(this->get_queue());

        float_t r_norm = 0;
        for (std::int64_t i = 0; i < n_; ++i) {
            float_t val = 0;
            for (std::int64_t j = 0; j < n_; ++j) {
                val += A_host_.at(i, j) * answer_host.at(j);
            }
            val -= b_host_.at(i);
            r_norm += val * val;
        }
        REQUIRE(r_norm < 1e-4);

        for (std::int64_t i = 0; i < n_; ++i) {
            check_val(x_host_.at(i), answer_host.at(i), (float_t)0.005, (float_t)0.005);
        }
    }

private:
    std::int64_t n_;
    ndarray<float_t, 2> A_host_;
    ndarray<float_t, 1> x_host_;
    ndarray<float_t, 1> b_host_;
};

using cg_solver_types = COMBINE_TYPES((float, double));

TEMPLATE_LIST_TEST_M(cg_solver_test,
                     "test with stable matrix",
                     "[cg-solver][gpu]",
                     cg_solver_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_input();
    this->test_cg_solver();
}

} // namespace oneapi::dal::backend::primitives::test
