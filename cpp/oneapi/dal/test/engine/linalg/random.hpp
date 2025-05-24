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

#pragma once

#include <random>
#include "oneapi/dal/test/engine/linalg/loops.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename T>
inline matrix<T> generate_uniform_matrix(const shape& s, T a, T b, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(a, b);

    auto m = matrix<T>::empty(s);
    enumerate_linear_mutable(m, [&](std::int64_t i, T& x) {
        x = T(uniform(rng));
    });

    return m;
}

/// Generates symmetric positive-definite matrix with diagonal dominance.
/// \f$\frac{1}{2}(A + A^T) + nE\f$, where $A$ is uniformly distributed matrix, \f$dim(A) = n\f$.
template <typename Float>
inline matrix<Float> generate_symmetric_positive_matrix(std::int64_t dim,
                                                        Float a,
                                                        Float b,
                                                        int seed) {
    const auto u = generate_uniform_matrix<Float>({ dim, dim }, a, b, seed);
    const auto ut = transpose(u);
    const auto c = multiply(Float(0.5), add(u, ut));
    return add(c, matrix<Float>::diag(dim, dim));
}

} // namespace oneapi::dal::test::engine::linalg
