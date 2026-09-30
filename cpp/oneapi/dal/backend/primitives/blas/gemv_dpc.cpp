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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include "oneapi/dal/backend/math_backend.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder ao>
sycl::event gemv(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 Float alpha,
                 Float beta,
                 const event_vector& deps) {
    ONEDAL_PROFILER_TASK(blas.gemv, queue);

    ONEDAL_ASSERT(a.has_data());
    ONEDAL_ASSERT(x.has_data());
    ONEDAL_ASSERT(y.has_mutable_data());
    const auto m = a.get_dimension(0);
    const auto n = a.get_dimension(1);
    const auto lda = a.get_leading_stride();
    ONEDAL_ASSERT(y.get_count() == m);
    ONEDAL_ASSERT(x.get_count() == n);

    // The order of matrix should be column major before it is transfered to gemv

    if constexpr (ao == ndorder::c) {
        ONEDAL_ASSERT(lda >= n);
        return mkl::blas::column_major::gemv(queue,
                               mkl::transpose::trans,
                               n,
                               m,
                               alpha,
                               a.get_data(),
                               lda,
                               x.get_data(),
                               std::int64_t(1),
                               beta,
                               y.get_mutable_data(),
                               std::int64_t(1),
                               deps);
    }
    else {
        ONEDAL_ASSERT(lda >= m);
        return mkl::blas::column_major::gemv(queue,
                               mkl::transpose::nontrans,
                               m,
                               n,
                               alpha,
                               a.get_data(),
                               lda,
                               x.get_data(),
                               std::int64_t(1),
                               beta,
                               y.get_mutable_data(),
                               std::int64_t(1),
                               deps);
    }
}

#define INSTANTIATE(F, ao)                                                    \
    template ONEDAL_EXPORT sycl::event gemv<F, ao>(sycl::queue & queue,       \
                                                   const ndview<F, 2, ao>& a, \
                                                   const ndview<F, 1>& x,     \
                                                   ndview<F, 1>& y,           \
                                                   F alpha,                   \
                                                   F beta,                    \
                                                   const event_vector& deps);

#define INSTANTIATE_FLOAT(ao) \
    INSTANTIATE(float, ao)    \
    INSTANTIATE(double, ao)

INSTANTIATE_FLOAT(ndorder::c);
INSTANTIATE_FLOAT(ndorder::f);

} // namespace oneapi::dal::backend::primitives
