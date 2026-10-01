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

#include "oneapi/dal/backend/primitives/blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/gemv.hpp"

namespace oneapi::dal::backend::primitives {

sycl::event optimize_gemv(sycl::queue& queue,
                          transpose transpose_a,
                          sparse_matrix_handle& a,
                          const event_vector& dependencies) {
    return mkl::sparse::optimize_gemv(queue,
                                      transpose_to_mkl(transpose_a),
                                      dal::detail::get_impl(a).get(),
                                      dependencies);
}

template <typename Float>
sycl::event gemv(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 const Float alpha,
                 const Float beta,
                 const event_vector& dependencies) {
    ONEDAL_ASSERT(x.has_data());
    ONEDAL_ASSERT(y.has_mutable_data());

    return mkl::sparse::gemv(queue,
                             transpose_to_mkl(transpose_a),
                             alpha,
                             dal::detail::get_impl(a).get(),
                             const_cast<Float*>(x.get_data()),
                             beta,
                             y.get_mutable_data(),
                             dependencies);
}

#define INSTANTIATE(F)                                                   \
    template ONEDAL_EXPORT sycl::event gemv<F>(sycl::queue & queue,      \
                                               transpose transpose_a,    \
                                               sparse_matrix_handle & a, \
                                               const ndview<F, 1>& x,    \
                                               ndview<F, 1>& y,          \
                                               const F alpha,            \
                                               const F beta,             \
                                               const event_vector& deps);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
