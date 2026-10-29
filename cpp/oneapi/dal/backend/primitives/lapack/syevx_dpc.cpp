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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/lapack/syevx.hpp"
#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

template <mkl::uplo uplo, typename Float>
sycl::event syevx(sycl::queue& queue,
                  ndview<Float, 2>& a,
                  std::int64_t il,
                  std::int64_t iu,
                  ndview<Float, 1>& eigenvalues,
                  ndview<Float, 2>& eigenvectors,
                  const event_vector& deps) {
    ONEDAL_PROFILER_TASK(syevx, queue);

    const std::int64_t n = a.get_dimension(1);
    const std::int64_t lda = a.get_stride(0);
    const std::int64_t ldz = eigenvectors.get_stride(0);
    ONEDAL_ASSERT(a.get_dimension(0) == n);
    ONEDAL_ASSERT(1 <= il && il <= iu && iu <= n);
    ONEDAL_ASSERT(eigenvalues.get_count() >= n);
    ONEDAL_ASSERT(eigenvectors.get_dimension(0) == iu - il + 1);
    ONEDAL_ASSERT(eigenvectors.get_dimension(1) == n);

    constexpr auto jobz = mkl::compz::vectors;
    constexpr auto range = mkl::rangev::indices;
    constexpr auto ul = ident_uplo(uplo);
    // abstol <= 0 lets the solver pick its default tolerance.
    const Float abstol = Float(0);

    const auto scratchpad_size = mkl::lapack::syevx_scratchpad_size<
        Float>(queue, jobz, range, ul, n, lda, Float(0), Float(0), il, iu, abstol, ldz);
    auto scratchpad =
        ndarray<Float, 1>::empty(queue, { scratchpad_size }, sycl::usm::alloc::device);
    auto found_count = ndarray<std::int64_t, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);

    auto event = mkl::lapack::syevx(queue,
                                    jobz,
                                    range,
                                    ul,
                                    n,
                                    a.get_mutable_data(),
                                    lda,
                                    Float(0),
                                    Float(0),
                                    il,
                                    iu,
                                    abstol,
                                    found_count.get_mutable_data(),
                                    eigenvalues.get_mutable_data(),
                                    eigenvectors.get_mutable_data(),
                                    ldz,
                                    scratchpad.get_mutable_data(),
                                    scratchpad_size,
                                    deps);
    // The scratchpad and the found count are local, so they must outlive the solver.
    event.wait_and_throw();
    return event;
}

#define INSTANTIATE(uplo, F)                                               \
    template ONEDAL_EXPORT sycl::event syevx<uplo, F>(sycl::queue & queue, \
                                                      ndview<F, 2> & a,    \
                                                      std::int64_t il,     \
                                                      std::int64_t iu,     \
                                                      ndview<F, 1>& w,     \
                                                      ndview<F, 2>& z,     \
                                                      const event_vector& deps);

#define INSTANTIATE_FLOAT(uplo) \
    INSTANTIATE(uplo, float)    \
    INSTANTIATE(uplo, double)

INSTANTIATE_FLOAT(mkl::uplo::upper)
INSTANTIATE_FLOAT(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
