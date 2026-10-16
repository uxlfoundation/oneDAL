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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"
#include "oneapi/dal/backend/primitives/lapack/misc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

namespace mkl = oneapi::dal::backend::math;

template <mkl::jobsvd jobu, mkl::jobsvd jobvt, typename Float>
sycl::event gesvd(sycl::queue& queue,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  ndview<Float, 2>& a,
                  std::int64_t lda,
                  ndview<Float, 1>& s,
                  ndview<Float, 2>& u,
                  std::int64_t ldu,
                  ndview<Float, 2>& vt,
                  std::int64_t ldvt,
                  const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
