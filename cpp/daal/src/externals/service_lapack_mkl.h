/* file: service_lapack_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_LAPACK_MKL_H__
#define __SERVICE_LAPACK_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>

#define __DAAL_MKLFN_CALL_LAPACK(f_name, f_args) f_name f_args;

#define __DAAL_MKLFN_CALL_RETURN_LAPACK(f_name, f_args) return f_name f_args;

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklLapack
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklLapack<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
    }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dgetrs,
                                 (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info));
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dgetrs,
                                 (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dpotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dpotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dpotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dpotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dpotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dpotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dormrq, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dormrq, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dtrtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dtrtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dpptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info));
    }

    static void xxpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dpptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, double * a, const DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            dgeqp3, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work, DAAL_INT lwork,
                        DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            dgeqp3, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, double * a, const DAAL_INT lda, const double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            dorgqr, ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            dorgqr, ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                       DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dgesvd, (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt,
                                          (MKL_INT *)(&ldvt), work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                        DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dgesvd, (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt,
                                          (MKL_INT *)(&ldvt), work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                       DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            dsyevd, (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            dsyevd, (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda,
                       const double * vl, const double * vu, const DAAL_INT * il, const DAAL_INT * iu, const double * abstol, DAAL_INT * m,
                       double * w, double * z, const DAAL_INT * ldz, DAAL_INT * isuppz, double * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                       const DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dsyevr, (jobz, range, uplo, (const MKL_INT *)n, a, (const MKL_INT *)lda, vl, vu, (const MKL_INT *)il,
                                          (const MKL_INT *)iu, abstol, (MKL_INT *)m, w, z, (const MKL_INT *)ldz, (MKL_INT *)isuppz, work,
                                          (const MKL_INT *)lwork, (MKL_INT *)iwork, (const MKL_INT *)liwork, (MKL_INT *)info));
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, double * a, const DAAL_INT * lda,
                        const double * vl, const double * vu, const DAAL_INT * il, const DAAL_INT * iu, const double * abstol, DAAL_INT * m,
                        double * w, double * z, const DAAL_INT * ldz, DAAL_INT * isuppz, double * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                        const DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dsyevr, (jobz, range, uplo, (const MKL_INT *)n, a, (const MKL_INT *)lda, vl, vu, (const MKL_INT *)il,
                                          (const MKL_INT *)iu, abstol, (MKL_INT *)m, w, z, (const MKL_INT *)ldz, (MKL_INT *)isuppz, work,
                                          (const MKL_INT *)lwork, (MKL_INT *)iwork, (const MKL_INT *)liwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(dormqr, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(dormqr, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xrscl(const DAAL_INT * n, const double * sa, double * sx, const DAAL_INT * incx)
    {
        __DAAL_MKLFN_CALL_LAPACK(drscl, ((MKL_INT *)n, sa, sx, (MKL_INT *)incx));
    }

    static void xxrscl(const DAAL_INT * n, const double * sa, double * sx, const DAAL_INT * incx)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(drscl, ((MKL_INT *)n, sa, sx, (MKL_INT *)incx));
        mkl_set_num_threads_local(old_nthr);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklLapack<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
    }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sgetrs,
                                 (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info));
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sgetrs,
                                 (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(spotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(spotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(spotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(spotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(spotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(spotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sormrq, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sormrq, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(strtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(strtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(spptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info));
    }

    static void xxpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(spptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, float * a, const DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            sgeqp3, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            sgeqp3, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, float * a, const DAAL_INT lda, const float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            sorgqr, ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            sorgqr, ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                       DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sgesvd, (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt,
                                          (MKL_INT *)(&ldvt), work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                        DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sgesvd, (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt,
                                          (MKL_INT *)(&ldvt), work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork, DAAL_INT * iwork,
                       DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(
            ssyevd, (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(
            ssyevd, (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda,
                       const float * vl, const float * vu, const DAAL_INT * il, const DAAL_INT * iu, const float * abstol, DAAL_INT * m, float * w,
                       float * z, const DAAL_INT * ldz, DAAL_INT * isuppz, float * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                       const DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(ssyevr, (jobz, range, uplo, (const MKL_INT *)n, a, (const MKL_INT *)lda, vl, vu, (const MKL_INT *)il,
                                          (const MKL_INT *)iu, abstol, (MKL_INT *)m, w, z, (const MKL_INT *)ldz, (MKL_INT *)isuppz, work,
                                          (const MKL_INT *)lwork, (MKL_INT *)iwork, (const MKL_INT *)liwork, (MKL_INT *)info));
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const DAAL_INT * n, float * a, const DAAL_INT * lda,
                        const float * vl, const float * vu, const DAAL_INT * il, const DAAL_INT * iu, const float * abstol, DAAL_INT * m, float * w,
                        float * z, const DAAL_INT * ldz, DAAL_INT * isuppz, float * work, const DAAL_INT * lwork, DAAL_INT * iwork,
                        const DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(ssyevr, (jobz, range, uplo, (const MKL_INT *)n, a, (const MKL_INT *)lda, vl, vu, (const MKL_INT *)il,
                                          (const MKL_INT *)iu, abstol, (MKL_INT *)m, w, z, (const MKL_INT *)ldz, (MKL_INT *)isuppz, work,
                                          (const MKL_INT *)lwork, (MKL_INT *)iwork, (const MKL_INT *)liwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL_LAPACK(sormqr, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(sormqr, (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work,
                                          (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xrscl(const DAAL_INT * n, const float * sa, float * sx, const DAAL_INT * incx)
    {
        __DAAL_MKLFN_CALL_LAPACK(srscl, ((MKL_INT *)n, sa, sx, (MKL_INT *)incx));
    }

    static void xxrscl(const DAAL_INT * n, const float * sa, float * sx, const DAAL_INT * incx)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_LAPACK(srscl, ((MKL_INT *)n, sa, sx, (MKL_INT *)incx));
        mkl_set_num_threads_local(old_nthr);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
