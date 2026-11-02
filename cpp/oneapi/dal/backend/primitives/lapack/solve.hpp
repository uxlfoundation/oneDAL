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

#include <optional>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL
namespace detail {

template <mkl::uplo uplo, typename Float, ndorder layout>
std::int64_t potrf_scratchpad_size(sycl::queue& queue, const ndview<Float, 2, layout>& x);

template <mkl::uplo uplo, typename Float, ndorder layout>
sycl::event potrf_factorization(sycl::queue& queue,
                                ndview<Float, 2, layout>& x,
                                array<Float>& scratchpad,
                                const event_vector& deps);

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
std::int64_t potrs_scratchpad_size(sycl::queue& queue,
                                   const ndview<Float, 2, xlayout>& x,
                                   const ndview<Float, 2, ylayout>& y);

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event potrs_solution(sycl::queue& queue,
                           ndview<Float, 2, xlayout>& x,
                           ndview<Float, 2, ylayout>& y,
                           array<Float>& scratchpad,
                           const event_vector& deps);

} // namespace detail

template <typename Float>
using opt_array = std::optional<array<Float>>;

template <mkl::uplo uplo, typename Float, ndorder layout>
array<Float> potrf_scratchpad(sycl::queue& queue,
                              const ndview<Float, 2, layout>& x,
                              const sycl::usm::alloc& alloc = sycl::usm::alloc::device);

template <mkl::uplo uplo, typename Float, ndorder layout>
sycl::event potrf_factorization(sycl::queue& queue,
                                ndview<Float, 2, layout>& x,
                                opt_array<Float>& scratchpad = {},
                                const event_vector& depenedincies = {});

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
array<Float> potrs_scratchpad(sycl::queue& queue,
                              const ndview<Float, 2, xlayout>& x,
                              const ndview<Float, 2, ylayout>& y,
                              const sycl::usm::alloc& alloc = sycl::usm::alloc::device);

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event potrs_solution(sycl::queue& queue,
                           ndview<Float, 2, xlayout>& x,
                           ndview<Float, 2, ylayout>& y,
                           opt_array<Float>& scratchpad = {},
                           const event_vector& depenedincies = {});

template <mkl::uplo uplo, bool beta, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event solve_system(sycl::queue& queue,
                         const ndview<Float, 2, xlayout>& xtx,
                         const ndview<Float, 2, ylayout>& xty,
                         ndview<Float, 2>& final_xtx,
                         ndview<Float, 2>& final_xty,
                         Float cutoff_factor = Float(1),
                         const event_vector& deps = {});

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
inline auto solve_system(sycl::queue& queue,
                         bool beta,
                         const ndview<Float, 2, xlayout>& xtx,
                         const ndview<Float, 2, ylayout>& xty,
                         ndview<Float, 2>& nxtx,
                         ndview<Float, 2>& nxty,
                         const event_vector& deps = {}) {
    if (beta)
        return solve_system<uplo, true>(queue, xtx, xty, nxtx, nxty, Float(1), deps);
    else
        return solve_system<uplo, false>(queue, xtx, xty, nxtx, nxty, Float(1), deps);
}

#endif

} // namespace oneapi::dal::backend::primitives
