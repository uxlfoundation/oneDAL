/* file: service_lapack_decl_mkl.h */
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

/*
//  Declarations of the MKL LAPACK routines that service_lapack_mkl.h calls.
//  mkl_lapack.h declares every LAPACK routine (about 50k lines) and costs
//  ~0.8 s of parsing in each DAAL translation unit; DAAL calls 34 of them.
//  library_version_info.cpp includes this file after <mkl.h>: a signature
//  that drifts from MKL is a "conflicting declaration" compile error, and a
//  name that MKL does not declare fails DAAL_CHECK_MKL_LAPACK_SUBSET.
*/

#ifndef __SERVICE_LAPACK_DECL_MKL_H__
#define __SERVICE_LAPACK_DECL_MKL_H__

#include <mkl_types.h>

// Every routine declared below.
// clang-format off
#define DAAL_MKL_LAPACK_SUBSET(X) \
    X(dgeqp3) X(dgeqrf) X(dgerqf) X(dgesvd) X(dgetrf) X(dgetrs) X(dorgqr) X(dormqr) X(dormrq) \
    X(dpotrf) X(dpotri) X(dpotrs) X(dpptrf) X(drscl) X(dsyevd) X(dsyevr) X(dtrtrs) \
    X(sgeqp3) X(sgeqrf) X(sgerqf) X(sgesvd) X(sgetrf) X(sgetrs) X(sorgqr) X(sormqr) X(sormrq) \
    X(spotrf) X(spotri) X(spotrs) X(spptrf) X(srscl) X(ssyevd) X(ssyevr) X(strtrs)
// clang-format on

#ifdef DAAL_CHECK_MKL_LAPACK_SUBSET
    // Set by library_version_info.cpp after <mkl.h>: each name must already be declared by MKL.
    #define __DAAL_MKL_LAPACK_DECLARED(f) static_assert(sizeof(&::f) > 0, #f " is not declared by <mkl.h>");
DAAL_MKL_LAPACK_SUBSET(__DAAL_MKL_LAPACK_DECLARED)
    #undef __DAAL_MKL_LAPACK_DECLARED
#endif

extern "C"
{
    void dgeqp3(const MKL_INT * m, const MKL_INT * n, double * a, const MKL_INT * lda, MKL_INT * jpvt, double * tau, double * work,
                const MKL_INT * lwork, MKL_INT * info) noexcept;
    void dgeqrf(const MKL_INT * m, const MKL_INT * n, double * a, const MKL_INT * lda, double * tau, double * work, const MKL_INT * lwork,
                MKL_INT * info) noexcept;
    void dgerqf(const MKL_INT * m, const MKL_INT * n, double * a, const MKL_INT * lda, double * tau, double * work, const MKL_INT * lwork,
                MKL_INT * info) noexcept;
    void dgesvd(const char * jobu, const char * jobvt, const MKL_INT * m, const MKL_INT * n, double * a, const MKL_INT * lda, double * s, double * u,
                const MKL_INT * ldu, double * vt, const MKL_INT * ldvt, double * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void dgetrf(const MKL_INT * m, const MKL_INT * n, double * a, const MKL_INT * lda, MKL_INT * ipiv, MKL_INT * info) noexcept;
    void dgetrs(const char * trans, const MKL_INT * n, const MKL_INT * nrhs, const double * a, const MKL_INT * lda, const MKL_INT * ipiv, double * b,
                const MKL_INT * ldb, MKL_INT * info) noexcept;
    void dorgqr(const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, double * a, const MKL_INT * lda, const double * tau, double * work,
                const MKL_INT * lwork, MKL_INT * info) noexcept;
    void dormqr(const char * side, const char * trans, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const double * a, const MKL_INT * lda,
                const double * tau, double * c, const MKL_INT * ldc, double * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void dormrq(const char * side, const char * trans, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const double * a, const MKL_INT * lda,
                const double * tau, double * c, const MKL_INT * ldc, double * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void dpotrf(const char * uplo, const MKL_INT * n, double * a, const MKL_INT * lda, MKL_INT * info) noexcept;
    void dpotri(const char * uplo, const MKL_INT * n, double * a, const MKL_INT * lda, MKL_INT * info) noexcept;
    void dpotrs(const char * uplo, const MKL_INT * n, const MKL_INT * nrhs, const double * a, const MKL_INT * lda, double * b, const MKL_INT * ldb,
                MKL_INT * info) noexcept;
    void dpptrf(const char * uplo, const MKL_INT * n, double * ap, MKL_INT * info) noexcept;
    void drscl(const MKL_INT * n, const double * sa, double * sx, const MKL_INT * incx) noexcept;
    void dsyevd(const char * jobz, const char * uplo, const MKL_INT * n, double * a, const MKL_INT * lda, double * w, double * work,
                const MKL_INT * lwork, MKL_INT * iwork, const MKL_INT * liwork, MKL_INT * info) noexcept;
    void dsyevr(const char * jobz, const char * range, const char * uplo, const MKL_INT * n, double * a, const MKL_INT * lda, const double * vl,
                const double * vu, const MKL_INT * il, const MKL_INT * iu, const double * abstol, MKL_INT * m, double * w, double * z,
                const MKL_INT * ldz, MKL_INT * isuppz, double * work, const MKL_INT * lwork, MKL_INT * iwork, const MKL_INT * liwork,
                MKL_INT * info) noexcept;
    void dtrtrs(const char * uplo, const char * trans, const char * diag, const MKL_INT * n, const MKL_INT * nrhs, const double * a,
                const MKL_INT * lda, double * b, const MKL_INT * ldb, MKL_INT * info) noexcept;
    void sgeqp3(const MKL_INT * m, const MKL_INT * n, float * a, const MKL_INT * lda, MKL_INT * jpvt, float * tau, float * work,
                const MKL_INT * lwork, MKL_INT * info) noexcept;
    void sgeqrf(const MKL_INT * m, const MKL_INT * n, float * a, const MKL_INT * lda, float * tau, float * work, const MKL_INT * lwork,
                MKL_INT * info) noexcept;
    void sgerqf(const MKL_INT * m, const MKL_INT * n, float * a, const MKL_INT * lda, float * tau, float * work, const MKL_INT * lwork,
                MKL_INT * info) noexcept;
    void sgesvd(const char * jobu, const char * jobvt, const MKL_INT * m, const MKL_INT * n, float * a, const MKL_INT * lda, float * s, float * u,
                const MKL_INT * ldu, float * vt, const MKL_INT * ldvt, float * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void sgetrf(const MKL_INT * m, const MKL_INT * n, float * a, const MKL_INT * lda, MKL_INT * ipiv, MKL_INT * info) noexcept;
    void sgetrs(const char * trans, const MKL_INT * n, const MKL_INT * nrhs, const float * a, const MKL_INT * lda, const MKL_INT * ipiv, float * b,
                const MKL_INT * ldb, MKL_INT * info) noexcept;
    void sorgqr(const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, float * a, const MKL_INT * lda, const float * tau, float * work,
                const MKL_INT * lwork, MKL_INT * info) noexcept;
    void sormqr(const char * side, const char * trans, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const float * a, const MKL_INT * lda,
                const float * tau, float * c, const MKL_INT * ldc, float * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void sormrq(const char * side, const char * trans, const MKL_INT * m, const MKL_INT * n, const MKL_INT * k, const float * a, const MKL_INT * lda,
                const float * tau, float * c, const MKL_INT * ldc, float * work, const MKL_INT * lwork, MKL_INT * info) noexcept;
    void spotrf(const char * uplo, const MKL_INT * n, float * a, const MKL_INT * lda, MKL_INT * info) noexcept;
    void spotri(const char * uplo, const MKL_INT * n, float * a, const MKL_INT * lda, MKL_INT * info) noexcept;
    void spotrs(const char * uplo, const MKL_INT * n, const MKL_INT * nrhs, const float * a, const MKL_INT * lda, float * b, const MKL_INT * ldb,
                MKL_INT * info) noexcept;
    void spptrf(const char * uplo, const MKL_INT * n, float * ap, MKL_INT * info) noexcept;
    void srscl(const MKL_INT * n, const float * sa, float * sx, const MKL_INT * incx) noexcept;
    void ssyevd(const char * jobz, const char * uplo, const MKL_INT * n, float * a, const MKL_INT * lda, float * w, float * work,
                const MKL_INT * lwork, MKL_INT * iwork, const MKL_INT * liwork, MKL_INT * info) noexcept;
    void ssyevr(const char * jobz, const char * range, const char * uplo, const MKL_INT * n, float * a, const MKL_INT * lda, const float * vl,
                const float * vu, const MKL_INT * il, const MKL_INT * iu, const float * abstol, MKL_INT * m, float * w, float * z,
                const MKL_INT * ldz, MKL_INT * isuppz, float * work, const MKL_INT * lwork, MKL_INT * iwork, const MKL_INT * liwork,
                MKL_INT * info) noexcept;
    void strtrs(const char * uplo, const char * trans, const char * diag, const MKL_INT * n, const MKL_INT * nrhs, const float * a,
                const MKL_INT * lda, float * b, const MKL_INT * ldb, MKL_INT * info) noexcept;
}

#endif
