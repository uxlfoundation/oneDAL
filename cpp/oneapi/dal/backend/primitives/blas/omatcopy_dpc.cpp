/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#include "oneapi/dal/backend/primitives/blas/omatcopy.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder src_order, ndorder dst_order>
sycl::event omatcopy(sycl::queue& queue,
                     const ndview<Float, 2, src_order>& src,
                     ndview<Float, 2, dst_order>& dst,
                     Float alpha,
                     const event_vector& deps) {
    ONEDAL_PROFILER_TASK(blas.omatcopy, queue);

    ONEDAL_ASSERT(src.get_shape() == dst.get_shape());
    ONEDAL_ASSERT(dst.has_mutable_data());

    constexpr auto trans = (src_order == dst_order) ? mkl::transpose::nontrans
                                                    : mkl::transpose::trans;

    // MKL omatcopy uses column-major layout by default.
    // For column-major (F-order) source, m and n map directly to dimensions.
    // For row-major (C-order) source, we swap m and n to account for the layout difference.
    std::int64_t mkl_m, mkl_n;
    if constexpr (src_order == ndorder::f) {
        mkl_m = src.get_dimension(0);
        mkl_n = src.get_dimension(1);
    }
    else {
        mkl_m = src.get_dimension(1);
        mkl_n = src.get_dimension(0);
    }

    return mkl::blas::omatcopy(queue,
                               trans,
                               mkl_m,
                               mkl_n,
                               alpha,
                               src.get_data(),
                               src.get_leading_stride(),
                               dst.get_mutable_data(),
                               dst.get_leading_stride(),
                               deps);
}

#define INSTANTIATE(F, src_o, dst_o)                            \
    template ONEDAL_EXPORT sycl::event omatcopy<F, src_o, dst_o>( \
        sycl::queue & queue,                                    \
        const ndview<F, 2, src_o>& src,                         \
        ndview<F, 2, dst_o>& dst,                               \
        F alpha,                                                \
        const event_vector& deps);

#define INSTANTIATE_FLOAT(src_o, dst_o) \
    INSTANTIATE(float, src_o, dst_o)    \
    INSTANTIATE(double, src_o, dst_o)

INSTANTIATE_FLOAT(ndorder::c, ndorder::c)
INSTANTIATE_FLOAT(ndorder::c, ndorder::f)
INSTANTIATE_FLOAT(ndorder::f, ndorder::c)
INSTANTIATE_FLOAT(ndorder::f, ndorder::f)

} // namespace oneapi::dal::backend::primitives
