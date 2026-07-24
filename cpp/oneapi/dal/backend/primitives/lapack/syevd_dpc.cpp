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
#include "oneapi/dal/backend/primitives/lapack/syevd.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
static sycl::event syevd_wrapper(sycl::queue& queue,
                                 mkl::job jobz,
                                 mkl::uplo uplo,
                                 std::int64_t column_count,
                                 Float* data_ptr,
                                 std::int64_t lda,
                                 Float* eigenvalues,
                                 Float* scratchpad,
                                 std::int64_t scratchpad_size,
                                 const event_vector& deps) {
    ONEDAL_ASSERT(lda >= column_count);

    return mkl::lapack::syevd(queue,
                              jobz,
                              uplo,
                              column_count,
                              data_ptr,
                              lda,
                              eigenvalues,
                              scratchpad,
                              scratchpad_size,
                              deps);
}

template <mkl::job jobz, mkl::uplo uplo, typename Float>
sycl::event syevd(sycl::queue& queue,
                  std::int64_t column_count,
                  ndview<Float, 2>& a,
                  std::int64_t lda,
                  ndview<Float, 1>& eigenvalues,
                  const event_vector& deps) {
    constexpr auto job = ident_job(jobz);
    constexpr auto ul = ident_uplo(uplo);

    const auto scratchpad_size =
        mkl::lapack::syevd_scratchpad_size<Float>(queue, jobz, uplo, column_count, lda);
    auto scratchpad =
        ndarray<Float, 1>::empty(queue, { scratchpad_size }, sycl::usm::alloc::device);

    return syevd_wrapper(queue,
                         job,
                         ul,
                         column_count,
                         a.get_mutable_data(),
                         lda,
                         eigenvalues.get_mutable_data(),
                         scratchpad.get_mutable_data(),
                         scratchpad_size,
                         deps);
}

#define INSTANTIATE(jobz, uplo, F)                                               \
    template ONEDAL_EXPORT sycl::event syevd<jobz, uplo, F>(sycl::queue & queue, \
                                                            std::int64_t n,      \
                                                            ndview<F, 2>& a,     \
                                                            std::int64_t lda,    \
                                                            ndview<F, 1>& w,     \
                                                            const event_vector& deps);

#define INSTANTIATE_FLOAT(jobz, uplo) \
    INSTANTIATE(jobz, uplo, float)    \
    INSTANTIATE(jobz, uplo, double)

#define INSTANTIATE_JOB(uplo)                \
    INSTANTIATE_FLOAT(mkl::job::novec, uplo) \
    INSTANTIATE_FLOAT(mkl::job::vec, uplo)

INSTANTIATE_JOB(mkl::uplo::upper)
INSTANTIATE_JOB(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
