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

#pragma once

/// Single point where the SYCL math library behind oneDAL's device primitives
/// is chosen.
///
/// Two libraries implement the same oneMKL DPC++ interface that the BLAS,
/// LAPACK, sparse BLAS and RNG primitives are written against:
///
///  * oneMKL (`<oneapi/mkl.hpp>`, namespace `oneapi::mkl`) — the default. Its
///    device backend targets Intel GPUs only.
///  * oneMath (`<oneapi/math.hpp>`, namespace `oneapi::math`) — the
///    open-source implementation of the same specification. Besides the Intel
///    backends it dispatches to cuBLAS, cuSOLVER, cuSPARSE and cuRAND, so the
///    unmodified oneDAL device sources also run on NVIDIA GPUs once the
///    compiler is asked for an `nvptx64-nvidia-cuda` target.
///
/// Both are reached through the `oneapi::dal::backend::math` alias, so the
/// ~150 `mkl::` call sites stay untouched and the decision is made once, at
/// build time, by defining `ONEDAL_MATH_BACKEND_ONEMATH`. With Bazel that
/// define arrives from the `@onemath//:onemath_dpc` dependency selected by
/// `--dpc_math_backend=onemath`.
///
/// This header intentionally contains no oneDAL types, so it can be included
/// from `dal/detail` as well as from `dal/backend/primitives`.

#ifdef ONEDAL_DATA_PARALLEL

#ifdef ONEDAL_MATH_BACKEND_ONEMATH
#include <oneapi/math.hpp>
#else
#include <oneapi/mkl.hpp>
#endif

namespace oneapi::dal::backend {

#ifdef ONEDAL_MATH_BACKEND_ONEMATH
namespace math = ::oneapi::math;
#else
namespace math = ::oneapi::mkl;
#endif

} // namespace oneapi::dal::backend

#endif // ONEDAL_DATA_PARALLEL
