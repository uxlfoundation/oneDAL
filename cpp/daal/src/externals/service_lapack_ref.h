/* file: service_lapack_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//++
//  Template wrappers for LAPACK functions.
//--
*/

#ifndef __SERVICE_LAPACK_REF_H__
#define __SERVICE_LAPACK_REF_H__

#include "service_lapack_declar_ref.h"
#include "service_thread_declar_ref.h"

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct OpenBlasLapack
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct OpenBlasLapack<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        dgetrf_(m, n, a, lda, ipiv, info);
    }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgetrf_(m, n, a, lda, ipiv, info);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        dgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info) { dpotrf_(uplo, p, ata, ldata, info); }

    static void xxpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dpotrf_(uplo, p, ata, ldata, info);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        dpotrs_(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dpotrs_(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info) { dpotri_(uplo, p, ata, ldata, info); }

    static void xxpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dpotri_(uplo, p, ata, ldata, info);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        dgerqf_(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgerqf_(m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        dormrq_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dormrq_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        dtrtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dtrtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info) { dpptrf_(uplo, n, ap, info); }

    static void xxpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dpptrf_(uplo, n, ap, info);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, double * a, const DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work, DAAL_INT lwork,
                        DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, double * a, const DAAL_INT lda, const double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                       DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info);
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                        DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                       DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda,
                       const double * vl, const double * vu, const DAAL_INT * il, const DAAL_INT * iu, const double * abstol, DAAL_INT * m,
                       double * w, double * z, const DAAL_INT * ldz, DAAL_INT * isuppz, double * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                       const DAAL_INT * liwork, DAAL_INT * info)
    {
        dsyevr_(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda,
                        const double * vl, const double * vu, const DAAL_INT * il, const DAAL_INT * iu, const double * abstol, DAAL_INT * m,
                        double * w, double * z, const DAAL_INT * ldz, DAAL_INT * isuppz, double * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                        const DAAL_INT * liwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dsyevr_(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        dormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        dormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xrscl(const DAAL_INT * n, const double * sa, double * sx, const DAAL_INT * incx) { drscl_(n, sa, sx, incx); }

    static void xxrscl(const DAAL_INT * n, const double * sa, double * sx, const DAAL_INT * incx)
    {
        openblas_thread_setter ots(1);
        drscl_(n, sa, sx, incx);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct OpenBlasLapack<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info) { sgetrf_(m, n, a, lda, ipiv, info); }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgetrf_(m, n, a, lda, ipiv, info);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        sgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info) { spotrf_(uplo, p, ata, ldata, info); }

    static void xxpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        spotrf_(uplo, p, ata, ldata, info);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        spotrs_(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        spotrs_(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info) { spotri_(uplo, p, ata, ldata, info); }

    static void xxpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        spotri_(uplo, p, ata, ldata, info);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        sgerqf_(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgerqf_(m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        sormrq_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sormrq_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        strtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        strtrs_(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info) { spptrf_(uplo, n, ap, info); }

    static void xxpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        spptrf_(uplo, n, ap, info);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, float * a, const DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        sgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, float * a, const DAAL_INT lda, const float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                       DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info);
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                        DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork, DAAL_INT * iwork,
                       DAAL_INT * liwork, DAAL_INT * info)
    {
        ssyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        ssyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda,
                       const float * vl, const float * vu, const DAAL_INT * il, const DAAL_INT * iu, const float * abstol, DAAL_INT * m, float * w,
                       float * z, const DAAL_INT * ldz, DAAL_INT * isuppz, float * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                       const DAAL_INT * liwork, DAAL_INT * info)
    {
        ssyevr_(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda,
                        const float * vl, const float * vu, const DAAL_INT * il, const DAAL_INT * iu, const float * abstol, DAAL_INT * m, float * w,
                        float * z, const DAAL_INT * ldz, DAAL_INT * isuppz, float * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                        const DAAL_INT * liwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        ssyevr_(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        sormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        openblas_thread_setter ots(1);
        sormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xrscl(const DAAL_INT * n, const float * sa, float * sx, const DAAL_INT * incx) { srscl_(n, sa, sx, incx); }

    static void xxrscl(const DAAL_INT * n, const float * sa, float * sx, const DAAL_INT * incx)
    {
        openblas_thread_setter ots(1);
        srscl_(n, sa, sx, incx);
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
