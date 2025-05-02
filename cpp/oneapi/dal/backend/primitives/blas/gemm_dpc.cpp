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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder ao, ndorder bo, ndorder co>
sycl::event gemm(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 const ndview<Float, 2, bo>& b,
                 ndview<Float, 2, co>& c,
                 Float alpha,
                 Float beta,
                 const event_vector& deps) {
    ONEDAL_PROFILER_TASK(blas.gemm, queue);

    ONEDAL_ASSERT(a.get_dimension(0) == c.get_dimension(0));
    ONEDAL_ASSERT(a.get_dimension(1) == b.get_dimension(0));
    ONEDAL_ASSERT(b.get_dimension(1) == c.get_dimension(1));
    ONEDAL_ASSERT(c.has_mutable_data());

    constexpr bool is_c_trans = (co == ndorder::c);
    if constexpr (is_c_trans) {
        return mkl::blas::gemm(queue,
                               f_order_as_transposed(bo),
                               f_order_as_transposed(ao),
                               c.get_dimension(1),
                               c.get_dimension(0),
                               a.get_dimension(1),
                               alpha,
                               b.get_data(),
                               b.get_leading_stride(),
                               a.get_data(),
                               a.get_leading_stride(),
                               beta,
                               c.get_mutable_data(),
                               c.get_leading_stride(),
                               deps);
    }
    else {
        return mkl::blas::gemm(queue,
                               c_order_as_transposed(ao),
                               c_order_as_transposed(bo),
                               c.get_dimension(0),
                               c.get_dimension(1),
                               a.get_dimension(1),
                               alpha,
                               a.get_data(),
                               a.get_leading_stride(),
                               b.get_data(),
                               b.get_leading_stride(),
                               beta,
                               c.get_mutable_data(),
                               c.get_leading_stride(),
                               deps);
    }
}

#define INSTANTIATE(F, ao, bo, co)                                                    \
    template ONEDAL_EXPORT sycl::event gemm<F, ao, bo, co>(sycl::queue & queue,       \
                                                           const ndview<F, 2, ao>& a, \
                                                           const ndview<F, 2, bo>& b, \
                                                           ndview<F, 2, co>& c,       \
                                                           F alpha,                   \
                                                           F beta,                    \
                                                           const event_vector& deps);

#define INSTANTIATE_FLOAT(ao, bo, co) \
    INSTANTIATE(float, ao, bo, co)    \
    INSTANTIATE(double, ao, bo, co)

INSTANTIATE_FLOAT(ndorder::c, ndorder::c, ndorder::c)
INSTANTIATE_FLOAT(ndorder::c, ndorder::c, ndorder::f)
INSTANTIATE_FLOAT(ndorder::c, ndorder::f, ndorder::c)
INSTANTIATE_FLOAT(ndorder::c, ndorder::f, ndorder::f)
INSTANTIATE_FLOAT(ndorder::f, ndorder::c, ndorder::c)
INSTANTIATE_FLOAT(ndorder::f, ndorder::c, ndorder::f)
INSTANTIATE_FLOAT(ndorder::f, ndorder::f, ndorder::c)
INSTANTIATE_FLOAT(ndorder::f, ndorder::f, ndorder::f)

} // namespace oneapi::dal::backend::primitives
