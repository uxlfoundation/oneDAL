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
#include "oneapi/dal/backend/primitives/sparse_blas/gemm.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder bo, ndorder co>
sycl::event gemm(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 2, bo>& b,
                 ndview<Float, 2, co>& c,
                 const Float alpha,
                 const Float beta,
                 const event_vector& dependencies) {
    ONEDAL_ASSERT(b.get_dimension(1) == c.get_dimension(1));
    ONEDAL_ASSERT(c.has_mutable_data());

#ifdef ONEDAL_MATH_BACKEND_ONEMATH
    // See the note in gemv_dpc.cpp: oneMath's `spmm` is a different interface
    // generation, not a different spelling of this call.
    throw unimplemented(dal::detail::error_messages::sparse_blas_not_available_in_math_backend());
#else
    if (co == ndorder::c) {
        return mkl::sparse::gemm(queue,
                                 order_as_layout(co),
                                 transpose_to_mkl(transpose_a),
                                 f_order_as_transposed(bo),
                                 alpha,
                                 dal::detail::get_impl(a).get(),
                                 const_cast<Float*>(b.get_data()),
                                 b.get_dimension(1),
                                 b.get_leading_stride(),
                                 beta,
                                 c.get_mutable_data(),
                                 c.get_leading_stride(),
                                 dependencies);
    }
    else {
        return mkl::sparse::gemm(queue,
                                 order_as_layout(co),
                                 transpose_to_mkl(transpose_a),
                                 c_order_as_transposed(bo),
                                 alpha,
                                 dal::detail::get_impl(a).get(),
                                 const_cast<Float*>(b.get_data()),
                                 b.get_dimension(1),
                                 b.get_leading_stride(),
                                 beta,
                                 c.get_mutable_data(),
                                 c.get_leading_stride(),
                                 dependencies);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
#endif // ONEDAL_MATH_BACKEND_ONEMATH
}

#define INSTANTIATE(F, bo, co)                                                    \
    template ONEDAL_EXPORT sycl::event gemm<F, bo, co>(sycl::queue & queue,       \
                                                       transpose transpose_a,     \
                                                       sparse_matrix_handle & a,  \
                                                       const ndview<F, 2, bo>& b, \
                                                       ndview<F, 2, co>& c,       \
                                                       const F alpha,             \
                                                       const F beta,              \
                                                       const event_vector& deps);

#define INSTANTIATE_FLOAT(bo, co) \
    INSTANTIATE(float, bo, co)    \
    INSTANTIATE(double, bo, co)

INSTANTIATE_FLOAT(ndorder::c, ndorder::c)
INSTANTIATE_FLOAT(ndorder::c, ndorder::f)
INSTANTIATE_FLOAT(ndorder::f, ndorder::c)
INSTANTIATE_FLOAT(ndorder::f, ndorder::f)

} // namespace oneapi::dal::backend::primitives
