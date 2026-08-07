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
#include "oneapi/dal/backend/primitives/lapack/gesvd.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
static sycl::event gesvd_wrapper(sycl::queue& queue,
                                 mkl::jobsvd jobu,
                                 mkl::jobsvd jobvt,
                                 std::int64_t row_count,
                                 std::int64_t column_count,
                                 Float* data_ptr,
                                 std::int64_t lda,
                                 Float* S_ptr,
                                 Float* U_ptr,
                                 std::int64_t ldu,
                                 Float* V_T_ptr,
                                 std::int64_t ldvt,
                                 Float* scratchpad_ptr,
                                 std::int64_t scratchpad_size,
                                 const event_vector& deps) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(lda > 0);
    ONEDAL_ASSERT(ldu > 0);
    ONEDAL_ASSERT(ldvt > 0);
    ONEDAL_ASSERT(scratchpad_size > 0);
    return mkl::lapack::gesvd(queue,
                              jobu,
                              jobvt,
                              row_count,
                              column_count,
                              data_ptr,
                              lda,
                              S_ptr,
                              U_ptr,
                              ldu,
                              V_T_ptr,
                              ldvt,
                              scratchpad_ptr,
                              scratchpad_size,
                              deps);
}

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
                  const event_vector& deps) {
    ONEDAL_PROFILER_TASK(lapack_gesvd, queue);

    constexpr auto job_u = ident_jobsvd(jobu);
    constexpr auto job_vt = ident_jobsvd(jobvt);

    const auto scratchpad_size = mkl::lapack::gesvd_scratchpad_size<Float>(queue,
                                                                           job_u,
                                                                           job_vt,
                                                                           row_count,
                                                                           column_count,
                                                                           lda,
                                                                           ldu,
                                                                           ldvt);
    auto scratchpad =
        ndarray<Float, 1>::empty(queue, { scratchpad_size }, sycl::usm::alloc::device);
    auto scratchpad_ptr = scratchpad.get_mutable_data();
    return gesvd_wrapper(queue,
                         job_u,
                         job_vt,
                         row_count,
                         column_count,
                         a.get_mutable_data(),
                         lda,
                         s.get_mutable_data(),
                         u.get_mutable_data(),
                         ldu,
                         vt.get_mutable_data(),
                         ldvt,
                         scratchpad_ptr,
                         scratchpad_size,
                         deps);
}

#define INSTANTIATE(jobu, jobvt, F)                                               \
    template ONEDAL_EXPORT sycl::event gesvd<jobu, jobvt, F>(sycl::queue & queue, \
                                                             std::int64_t m,      \
                                                             std::int64_t n,      \
                                                             ndview<F, 2>& a,     \
                                                             std::int64_t lda,    \
                                                             ndview<F, 1>& s,     \
                                                             ndview<F, 2>& u,     \
                                                             std::int64_t ldu,    \
                                                             ndview<F, 2>& vt,    \
                                                             std::int64_t ldvt,   \
                                                             const event_vector& deps);

#define INSTANTIATE_FLOAT(jobu, jobvt) \
    INSTANTIATE(jobu, jobvt, float)    \
    INSTANTIATE(jobu, jobvt, double)

#define INSTANTIATE_JOB(jobvt)                        \
    INSTANTIATE_FLOAT(mkl::jobsvd::vectors, jobvt)    \
    INSTANTIATE_FLOAT(mkl::jobsvd::somevec, jobvt)    \
    INSTANTIATE_FLOAT(mkl::jobsvd::vectorsina, jobvt) \
    INSTANTIATE_FLOAT(mkl::jobsvd::novec, jobvt)

INSTANTIATE_JOB(mkl::jobsvd::vectors)
INSTANTIATE_JOB(mkl::jobsvd::somevec)
INSTANTIATE_JOB(mkl::jobsvd::novec)

} // namespace oneapi::dal::backend::primitives
