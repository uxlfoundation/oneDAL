/* file: service_lapack.h */
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
//  Template wrappers for LAPACK functions.
//--
*/

/* Note: this file is not auto-generated. These 'x'/'xx' functions are manually added here on an
as-needed basis, and are only used internally within the library so their signatures might not
match LAPACK's to every minutiae like passing pointers to scalars or passing them by value, or
having 'const' qualifiers or not. */

#ifndef __SERVICE_LAPACK_H__
#define __SERVICE_LAPACK_H__

#include "services/daal_defines.h"
#include "src/externals/service_dispatch.h"
#include "src/externals/service_memory.h"

#include "src/externals/config.h"

#define DAAL_CALL_LAPACK_CPU_FUNC(cpuId, fpType, funcName, ...) LapackInst<fpType, cpuId>::funcName(__VA_ARGS__)

#define DAAL_DISPATCH_LAPACK_BY_CPU(...) DAAL_EXPAND(DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_CALL_LAPACK_CPU_FUNC, __VA_ARGS__))

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl>
struct Lapack
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;

    static void xgetrf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, SizeType * ipiv, SizeType * info)
    {
        _impl<fpType, cpu>::xgetrf(m, n, a, lda, ipiv, info);
    }

    static void xxgetrf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, SizeType * ipiv, SizeType * info)
    {
        _impl<fpType, cpu>::xxgetrf(m, n, a, lda, ipiv, info);
    }

    static void xgetrs(char * trans, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, SizeType * ipiv, fpType * b, SizeType * ldb,
                       SizeType * info)
    {
        _impl<fpType, cpu>::xgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xxgetrs(char * trans, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, SizeType * ipiv, fpType * b, SizeType * ldb,
                        SizeType * info)
    {
        _impl<fpType, cpu>::xxgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xpotrf(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        _impl<fpType, cpu>::xpotrf(uplo, p, ata, ldata, info);
    }

    static void xxpotrf(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        _impl<fpType, cpu>::xxpotrf(uplo, p, ata, ldata, info);
    }

    static void xpotrs(char * uplo, SizeType * p, SizeType * ny, fpType * ata, SizeType * ldata, fpType * beta, SizeType * ldaty, SizeType * info)
    {
        _impl<fpType, cpu>::xpotrs(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xxpotrs(char * uplo, SizeType * p, SizeType * ny, fpType * ata, SizeType * ldata, fpType * beta, SizeType * ldaty, SizeType * info)
    {
        _impl<fpType, cpu>::xxpotrs(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        _impl<fpType, cpu>::xpotri(uplo, p, ata, ldata, info);
    }

    static void xxpotri(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        _impl<fpType, cpu>::xxpotri(uplo, p, ata, ldata, info);
    }

    static void xgerqf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, fpType * tau, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xgerqf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgerqf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, fpType * tau, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxgerqf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                       SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormrq(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                        SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                       SizeType * info)
    {
        _impl<fpType, cpu>::xtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                        SizeType * info)
    {
        _impl<fpType, cpu>::xxtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char * uplo, SizeType * n, fpType * ap, SizeType * info) { _impl<fpType, cpu>::xpptrf(uplo, n, ap, info); }

    static void xxpptrf(char * uplo, SizeType * n, fpType * ap, SizeType * info) { _impl<fpType, cpu>::xxpptrf(uplo, n, ap, info); }

    static void xgeqrf(SizeType m, SizeType n, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xgeqrf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgeqrf(SizeType m, SizeType n, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxgeqrf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xgeqp3(const SizeType m, const SizeType n, fpType * a, const SizeType lda, SizeType * jpvt, fpType * tau, fpType * work,
                       const SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xxgeqp3(SizeType m, SizeType n, fpType * a, SizeType lda, SizeType * jpvt, fpType * tau, fpType * work, SizeType lwork,
                        SizeType * info)
    {
        _impl<fpType, cpu>::xxgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xorgqr(const SizeType m, const SizeType n, const SizeType k, fpType * a, const SizeType lda, const fpType * tau, fpType * work,
                       const SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xxorgqr(SizeType m, SizeType n, SizeType k, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, SizeType m, SizeType n, fpType * a, SizeType lda, fpType * s, fpType * u, SizeType ldu, fpType * vt,
                       SizeType ldvt, fpType * work, SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xxgesvd(char jobu, char jobvt, SizeType m, SizeType n, fpType * a, SizeType lda, fpType * s, fpType * u, SizeType ldu, fpType * vt,
                        SizeType ldvt, fpType * work, SizeType lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    /* Note: 'syevd' and 'syevr' both compute symmetric eigenvalues, but they use different routines. 'syevd'
    is slower but more precise and should thus be preferred for situations in which small numerical inaccuracies
    have adverse side effects, whereas 'syevr' is faster and more suitable for general usage. */
    static void xsyevd(char * jobz, char * uplo, SizeType * n, fpType * a, SizeType * lda, fpType * w, fpType * work, SizeType * lwork,
                       SizeType * iwork, SizeType * liwork, SizeType * info)
    {
        _impl<fpType, cpu>::xsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xxsyevd(char * jobz, char * uplo, SizeType * n, fpType * a, SizeType * lda, fpType * w, fpType * work, SizeType * lwork,
                        SizeType * iwork, SizeType * liwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const SizeType * n, fpType * a, const SizeType * lda,
                       const fpType * vl, const fpType * vu, const SizeType * il, const SizeType * iu, const fpType * abstol, SizeType * m,
                       fpType * w, fpType * z, const SizeType * ldz, SizeType * isuppz, fpType * work, const SizeType * lwork, SizeType * iwork,
                       const SizeType * liwork, SizeType * info)
    {
        _impl<fpType, cpu>::xsyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const SizeType * n, fpType * a, const SizeType * lda,
                        const fpType * vl, const fpType * vu, const SizeType * il, const SizeType * iu, const fpType * abstol, SizeType * m,
                        fpType * w, fpType * z, const SizeType * ldz, SizeType * isuppz, fpType * work, const SizeType * lwork, SizeType * iwork,
                        const SizeType * liwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxsyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info);
    }

    static void xormqr(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                       SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormqr(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                        SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        _impl<fpType, cpu>::xxormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xrscl(const SizeType * n, const fpType * sa, fpType * sx, const SizeType * incx) { _impl<fpType, cpu>::xrscl(n, sa, sx, incx); }

    static void xxrscl(const SizeType * n, const fpType * sa, fpType * sx, const SizeType * incx) { _impl<fpType, cpu>::xxrscl(n, sa, sx, incx); }
};

template <typename fpType, CpuType cpu>
using LapackInst = Lapack<fpType, cpu, LapackBackend>;

template <typename fpType>
struct LapackAutoDispatch
{
    typedef typename LapackInst<fpType, DAAL_BASE_CPU>::SizeType SizeType;

    static void xgetrf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, SizeType * ipiv, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgetrf, m, n, a, lda, ipiv, info);
    }

    static void xxgetrf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, SizeType * ipiv, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgetrf, m, n, a, lda, ipiv, info);
    }

    static void xgetrs(char * trans, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, SizeType * ipiv, fpType * b, SizeType * ldb,
                       SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgetrs, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xxgetrs(char * trans, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, SizeType * ipiv, fpType * b, SizeType * ldb,
                        SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgetrs, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }

    static void xpotrf(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xpotrf, uplo, p, ata, ldata, info);
    }

    static void xxpotrf(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxpotrf, uplo, p, ata, ldata, info);
    }

    static void xpotrs(char * uplo, SizeType * p, SizeType * ny, fpType * ata, SizeType * ldata, fpType * beta, SizeType * ldaty, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xpotrs, uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xxpotrs(char * uplo, SizeType * p, SizeType * ny, fpType * ata, SizeType * ldata, fpType * beta, SizeType * ldaty, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxpotrs, uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xpotri, uplo, p, ata, ldata, info);
    }

    static void xxpotri(char * uplo, SizeType * p, fpType * ata, SizeType * ldata, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xpotri, uplo, p, ata, ldata, info);
    }

    static void xgerqf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, fpType * tau, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgerqf, m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgerqf(SizeType * m, SizeType * n, fpType * a, SizeType * lda, fpType * tau, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgerqf, m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                       SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xormrq, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormrq(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                        SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxormrq, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                       SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xtrtrs, uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, SizeType * n, SizeType * nrhs, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                        SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxtrtrs, uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char * uplo, SizeType * n, fpType * ap, SizeType * info) { DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xpptrf, uplo, n, ap, info); }

    static void xxpptrf(char * uplo, SizeType * n, fpType * ap, SizeType * info) { DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxpptrf, uplo, n, ap, info); }

    static void xgeqrf(SizeType m, SizeType n, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgeqrf, m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgeqrf(SizeType m, SizeType n, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgeqrf, m, n, a, lda, tau, work, lwork, info);
    }

    static void xgeqp3(const SizeType m, const SizeType n, fpType * a, const SizeType lda, SizeType * jpvt, fpType * tau, fpType * work,
                       const SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgeqp3, m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xxgeqp3(SizeType m, SizeType n, fpType * a, SizeType lda, SizeType * jpvt, fpType * tau, fpType * work, SizeType lwork,
                        SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgeqp3, m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xorgqr(const SizeType m, const SizeType n, const SizeType k, fpType * a, const SizeType lda, const fpType * tau, fpType * work,
                       const SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xorgqr, m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xxorgqr(SizeType m, SizeType n, SizeType k, fpType * a, SizeType lda, fpType * tau, fpType * work, SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxorgqr, m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, SizeType m, SizeType n, fpType * a, SizeType lda, fpType * s, fpType * u, SizeType ldu, fpType * vt,
                       SizeType ldvt, fpType * work, SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xgesvd, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xxgesvd(char jobu, char jobvt, SizeType m, SizeType n, fpType * a, SizeType lda, fpType * s, fpType * u, SizeType ldu, fpType * vt,
                        SizeType ldvt, fpType * work, SizeType lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxgesvd, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xsyevd(char * jobz, char * uplo, SizeType * n, fpType * a, SizeType * lda, fpType * w, fpType * work, SizeType * lwork,
                       SizeType * iwork, SizeType * liwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xsyevd, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xxsyevd(char * jobz, char * uplo, SizeType * n, fpType * a, SizeType * lda, fpType * w, fpType * work, SizeType * lwork,
                        SizeType * iwork, SizeType * liwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxsyevd, jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xsyevr(const char * jobz, const char * range, const char * uplo, const SizeType * n, fpType * a, const SizeType * lda,
                       const fpType * vl, const fpType * vu, const SizeType * il, const SizeType * iu, const fpType * abstol, SizeType * m,
                       fpType * w, fpType * z, const SizeType * ldz, SizeType * isuppz, fpType * work, const SizeType * lwork, SizeType * iwork,
                       const SizeType * liwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xsyevr, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork,
                                    liwork, info);
    }

    static void xxsyevr(const char * jobz, const char * range, const char * uplo, const SizeType * n, fpType * a, const SizeType * lda,
                        const fpType * vl, const fpType * vu, const SizeType * il, const SizeType * iu, const fpType * abstol, SizeType * m,
                        fpType * w, fpType * z, const SizeType * ldz, SizeType * isuppz, fpType * work, const SizeType * lwork, SizeType * iwork,
                        const SizeType * liwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxsyevr, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work, lwork, iwork,
                                    liwork, info);
    }

    static void xormqr(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                       SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xormqr, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormqr(char * side, char * trans, SizeType * m, SizeType * n, SizeType * k, fpType * a, SizeType * lda, fpType * tau, fpType * c,
                        SizeType * ldc, fpType * work, SizeType * lwork, SizeType * info)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxormqr, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xrscl(SizeType * n, const fpType * sa, fpType * sx, const SizeType * incx)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xrscl, n, sa, sx, incx);
    }

    static void xxrscl(SizeType * n, const fpType * sa, fpType * sx, const SizeType * incx)
    {
        DAAL_DISPATCH_LAPACK_BY_CPU(fpType, xxrscl, n, sa, sx, incx);
    }
};

} // namespace internal
} // namespace daal

#undef DAAL_CALL_LAPACK_CPU_FUNC
#undef DAAL_DISPATCH_LAPACK_BY_CPU

#endif
