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
#include "oneapi/dal/backend/primitives/blas/syrk.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include "oneapi/dal/backend/math_backend.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
static sycl::event syrk_wrapper(sycl::queue& queue,
                                mkl::uplo uplo,
                                mkl::transpose trans,
                                std::int64_t n,
                                std::int64_t k,
                                Float alpha,
                                const Float* a,
                                std::int64_t lda,
                                Float beta,
                                Float* c,
                                std::int64_t ldc,
                                const event_vector& deps) {
    [[maybe_unused]] const bool is_trans = (trans == mkl::transpose::trans);
    ONEDAL_ASSERT(ldc >= n);
    ONEDAL_ASSERT(is_trans || lda >= n);
    ONEDAL_ASSERT(!is_trans || lda >= k);

    return mkl::blas::column_major::syrk(queue,
                                         uplo,
                                         trans,
                                         n,
                                         k,
                                         alpha,
                                         a,
                                         lda,
                                         beta,
                                         c,
                                         ldc,
                                         deps);
}

template <mkl::uplo uplo, typename Float, ndorder ao>
sycl::event syrk(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 ndview<Float, 2>& c,
                 Float alpha,
                 Float beta,
                 const event_vector& deps) {
    ONEDAL_PROFILER_TASK(blas.syrk, queue);

    ONEDAL_ASSERT(a.get_dimension(1) == c.get_dimension(0));
    ONEDAL_ASSERT(c.get_dimension(0) == c.get_dimension(1));
    ONEDAL_ASSERT(c.has_mutable_data());

    const auto nd = c.get_dimension(0);
    const auto kd = a.get_dimension(0);
    const auto* const a_ptr = a.get_data();
    auto* const c_ptr = c.get_mutable_data();
    const auto a_str = a.get_leading_stride();
    const auto c_str = c.get_leading_stride();
    constexpr auto tr = f_order_as_transposed(ao);
    constexpr auto ul = ident_uplo(uplo);
    return syrk_wrapper(queue, ul, tr, nd, kd, alpha, a_ptr, a_str, beta, c_ptr, c_str, deps);
}

#define INSTANTIATE(ul, F, ao)                                                    \
    template ONEDAL_EXPORT sycl::event syrk<ul, F, ao>(sycl::queue & queue,       \
                                                       const ndview<F, 2, ao>& a, \
                                                       ndview<F, 2>& c,           \
                                                       F alpha,                   \
                                                       F beta,                    \
                                                       const event_vector& deps);

#define INSTANTIATE_FLOAT(ul, ao) \
    INSTANTIATE(ul, float, ao)    \
    INSTANTIATE(ul, double, ao)

#define INSTANTIATE_UPLO(ao)                \
    INSTANTIATE_FLOAT(mkl::uplo::upper, ao) \
    INSTANTIATE_FLOAT(mkl::uplo::lower, ao)

INSTANTIATE_UPLO(ndorder::c)
INSTANTIATE_UPLO(ndorder::f)

} // namespace oneapi::dal::backend::primitives
