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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/math_backend.hpp"

namespace oneapi::dal::backend::primitives {

namespace mkl = oneapi::dal::backend::math;

#ifdef ONEDAL_DATA_PARALLEL
/// Convert oneDAL `ndorder` to oneMKL `layout`
inline constexpr mkl::layout order_as_layout(ndorder order) {
    return (order == ndorder::c) ? mkl::layout::R /* row-major */
                                 : mkl::layout::C /* column-major */;
}

/// Convert oneDAL `transpose` to oneMKL `transpose`
inline constexpr mkl::transpose transpose_to_mkl(primitives::transpose trans) {
    return (trans == primitives::transpose::trans) ? mkl::transpose::trans
                                                   : mkl::transpose::nontrans;
}

inline constexpr mkl::transpose f_order_as_transposed(ndorder order) {
    return (order == ndorder::f) ? mkl::transpose::trans : mkl::transpose::nontrans;
}

inline constexpr mkl::transpose c_order_as_transposed(ndorder order) {
    return (order == ndorder::c) ? mkl::transpose::trans : mkl::transpose::nontrans;
}

inline constexpr mkl::uplo flip_uplo(mkl::uplo order) {
    constexpr auto upper = mkl::uplo::upper;
    constexpr auto lower = mkl::uplo::lower;
    return (order == upper) ? lower : upper;
}

inline constexpr mkl::uplo ident_uplo(mkl::uplo order) {
    constexpr auto upper = mkl::uplo::upper;
    constexpr auto lower = mkl::uplo::lower;
    return (order == upper) ? upper : lower;
}
#endif
} // namespace oneapi::dal::backend::primitives
