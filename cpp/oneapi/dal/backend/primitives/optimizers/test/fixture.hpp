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

#pragma once

#include <type_traits>
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::backend::primitives::test {

// f(x) = 1/2 x^t A x - b^t x
// df / dx = Ax - b
// df / d^2x = A
template <typename Float>
class quadratic_function : public base_function<Float> {
public:
    quadratic_function(sycl::queue& q, const ndview<Float, 2>& A, const ndview<Float, 1>& b)
            : q_(q),
              n_(A.get_dimension(0)),
              A_(A),
              b_(b),
              hessp_(q, A) {
        ONEDAL_ASSERT(A.get_dimension(1) == n_);
        ONEDAL_ASSERT(b.get_dimension(0) == n_);
        gradient_ = ndarray<Float, 1>::empty(q, { n_ }, sycl::usm::alloc::device);
        tmp_ = ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device);
    }
    Float get_value() final {
        return value_;
    }

    ndview<Float, 1>& get_gradient() final {
        return gradient_;
    }

    base_matrix_operator<Float>& get_hessian_product() final {
        return hessp_;
    }

    event_vector update_x(const ndview<Float, 1>& x,
                          bool need_hessp = false,
                          const event_vector& deps = {}) final {
        constexpr Float zero(0), one(1);
        auto fill_gradient_event = fill<Float>(q_, gradient_, zero, deps);
        auto fill_value_event = fill<Float>(q_, tmp_, zero, deps);
        auto gemv_event = gemv(q_, A_, x, gradient_, one, zero, { fill_gradient_event }); // Ax
        Float tmp_host = 0;
        auto xtax_event = dot_product(q_,
                                      gradient_,
                                      x,
                                      tmp_.get_mutable_data(),
                                      &tmp_host,
                                      { gemv_event }); //x^tAx
        auto btx_event =
            dot_product(q_, b_, x, tmp_.get_mutable_data(), &value_, { xtax_event }); // b^t x
        const auto kernel_minus = [=](const Float a, const Float b) -> Float {
            return a - b;
        };
        auto bias_event = element_wise(q_, kernel_minus, gradient_, b_, gradient_, { xtax_event });

        btx_event.wait_and_throw();
        value_ = -value_ + tmp_host / 2; // 1/2 x^t A x - b^t x

        return { bias_event };
    }

private:
    sycl::queue q_;
    const std::int64_t n_;
    const ndview<Float, 2> A_;
    const ndview<Float, 1> b_;
    Float value_;
    ndarray<Float, 1> tmp_;
    ndarray<Float, 1> gradient_;
    linear_matrix_operator<Float> hessp_;
};

template <typename Float>
void check_val(const Float real, const Float expected, const Float rtol, const Float atol) {
    REQUIRE(abs(real - expected) < atol);
    REQUIRE(abs(real - expected) / std::max(std::abs(expected), Float(1.0)) < rtol);
}

template <typename Float>
void gram_schmidt(ndview<Float, 2>& A) {
    const std::int64_t n = A.get_dimension(0);
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < i; ++j) {
            Float res = 0;
            for (std::int64_t k = 0; k < n; ++k) {
                res += A.at(i, k) * A.at(j, k);
            }
            for (std::int64_t k = 0; k < n; ++k) {
                A.at(i, k) -= res * A.at(j, k);
            }
        }
        Float norm = 0;
        for (std::int64_t k = 0; k < n; ++k) {
            norm += A.at(i, k) * A.at(i, k);
        }
        norm = sqrt(norm);
        for (std::int64_t k = 0; k < n; ++k) {
            A.at(i, k) /= norm;
        }
    }
}

template <typename Float>
void create_stable_matrix(sycl::queue& queue,
                          ndview<Float, 2>& A,
                          Float bottom_eig = 1.0,
                          Float top_eig = 2.0) {
    const std::int64_t n = A.get_dimension(0);
    ONEDAL_ASSERT(A.get_dimension(1) == n);
    auto J = ndarray<Float, 2>::empty(queue, { n, n }, sycl::usm::alloc::host);
    auto eigen_values = ndarray<Float, 1>::empty(queue, { n }, sycl::usm::alloc::host);
    primitives::host_engine eng(2007 + n);

    primitives::uniform<Float>(n * n, J.get_mutable_data(), eng, -1.0, 1.0);
    primitives::uniform<Float>(n, eigen_values.get_mutable_data(), eng, bottom_eig, top_eig);

    // orthogonalize matrix J
    gram_schmidt(J);

    // A = J D J^T so matrix A is symmetric with eigen values equal to diagonal elements of D
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            A.at(i, j) = 0;
            for (std::int64_t k = 0; k < n; ++k) {
                A.at(i, j) += J.at(i, k) * J.at(j, k) * eigen_values.at(k);
            }
        }
    }
}
} // namespace oneapi::dal::backend::primitives::test
