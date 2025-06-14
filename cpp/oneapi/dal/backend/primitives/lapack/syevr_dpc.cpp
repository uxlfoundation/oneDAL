// #pragma once

// #include "oneapi/dal/backend/primitives/ndarray.hpp"
// #include "oneapi/dal/backend/primitives/blas/misc.hpp"
// #include "oneapi/dal/backend/primitives/lapack/misc.hpp"

// namespace oneapi::dal::backend::primitives {

// #ifdef ONEDAL_DATA_PARALLEL

// #include <oneapi/mkl/lapack.hpp>
// #include "oneapi/dal/detail/profiler.hpp"

// namespace mkl = oneapi::mkl;

// template <typename Float>
// static sycl::event syevr_wrapper(sycl::queue& queue,
//                                  mkl::job jobz,
//                                  mkl::range range,
//                                  mkl::uplo uplo,
//                                  std::int64_t n,
//                                  Float* a,
//                                  std::int64_t lda,
//                                  Float vl,
//                                  Float vu,
//                                  std::int64_t il,
//                                  std::int64_t iu,
//                                  Float* w,
//                                  Float* z,
//                                  std::int64_t ldz,
//                                  std::int64_t* isuppz,
//                                  std::int64_t& m,
//                                  Float* scratchpad,
//                                  std::int64_t scratchpad_size,
//                                  const event_vector& deps) {
//     ONEDAL_ASSERT(lda >= n);
//     return mkl::lapack::syevr(queue,
//                               jobz,
//                               range,
//                               uplo,
//                               n,
//                               a,
//                               lda,
//                               vl,
//                               vu,
//                               il,
//                               iu,
//                               m,
//                               w,
//                               z,
//                               ldz,
//                               isuppz,
//                               scratchpad,
//                               scratchpad_size,
//                               deps);
// }

// template <mkl::job jobz, mkl::range range, mkl::uplo uplo, typename Float>
// sycl::event syevr(sycl::queue& queue,
//                   std::int64_t column_count,
//                   ndview<Float, 2>& a,
//                   std::int64_t lda,
//                   ndview<Float, 1>& eigenvalues,
//                   ndview<Float, 2>& eigenvectors,
//                   ndview<std::int64_t, 1>& isuppz,
//                   const event_vector& deps) {
//     constexpr auto job = ident_job(jobz);
//     constexpr auto rng = ident_range(range);
//     constexpr auto ul = ident_uplo(uplo);

//     const Float vl = Float{0};  // not used when range = all
//     const Float vu = Float{0};
//     const std::int64_t il = 0;
//     const std::int64_t iu = 0;

//     std::int64_t m = 0;
//     const std::int64_t ldz = column_count;

//     const auto scratchpad_size = mkl::lapack::syevr_scratchpad_size<Float>(queue,
//                                                                            jobz,
//                                                                            range,
//                                                                            uplo,
//                                                                            column_count,
//                                                                            lda,
//                                                                            vl,
//                                                                            vu,
//                                                                            il,
//                                                                            iu,
//                                                                            ldz);

//     auto scratchpad =
//         ndarray<Float, 1>::empty(queue, { scratchpad_size }, sycl::usm::alloc::device);

//     return syevr_wrapper(queue,
//                          job,
//                          rng,
//                          ul,
//                          column_count,
//                          a.get_mutable_data(),
//                          lda,
//                          vl,
//                          vu,
//                          il,
//                          iu,
//                          eigenvalues.get_mutable_data(),
//                          eigenvectors.get_mutable_data(),
//                          ldz,
//                          isuppz.get_mutable_data(),
//                          m,
//                          scratchpad.get_mutable_data(),
//                          scratchpad_size,
//                          deps);
// }

// #define INSTANTIATE(jobz, range, uplo, F)                                           \
//     template ONEDAL_EXPORT sycl::event syevr<jobz, range, uplo, F>(                 \
//         sycl::queue& queue,                                                        \
//         std::int64_t n,                                                            \
//         ndview<F, 2>& a,                                                           \
//         std::int64_t lda,                                                          \
//         ndview<F, 1>& w,                                                           \
//         ndview<F, 2>& z,                                                           \
//         ndview<std::int64_t, 1>& isuppz,                                           \
//         const event_vector& deps);

// #define INSTANTIATE_FLOAT(jobz, range, uplo) \
//     INSTANTIATE(jobz, range, uplo, float)    \
//     INSTANTIATE(jobz, range, uplo, double)

// #define INSTANTIATE_ALL(uplo)                                  \
//     INSTANTIATE_FLOAT(mkl::job::novec, mkl::range::all, uplo)  \
//     INSTANTIATE_FLOAT(mkl::job::vec, mkl::range::all, uplo)

// INSTANTIATE_ALL(mkl::uplo::upper)
// INSTANTIATE_ALL(mkl::uplo::lower)

// #endif

// } // namespace oneapi::dal::backend::primitives
