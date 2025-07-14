/* file: service_blas_ref.h */
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
//  Template wrappers for common BLAS functions.
//--
*/

#ifndef __SERVICE_BLAS_REF_H__
#define __SERVICE_BLAS_REF_H__

#include "service_blas_declar_ref.h"
#include "service_thread_declar_ref.h"

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct OpenBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct OpenBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        dsyrk_(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                       const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        openblas_thread_setter ots(1);
        dsyrk_(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                     const DAAL_INT * lda)
    {
        dsyr_(uplo, n, alpha, x, incx, a, lda);
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                      const DAAL_INT * lda)
    {
        openblas_thread_setter ots(1);
        dsyr_(uplo, n, alpha, x, incx, a, lda);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                      const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                      const DAAL_INT * ldaty)
    {
        dgemm_(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                       const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                       const DAAL_INT * ldaty)
    {
        openblas_thread_setter ots(1);
        dgemm_(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        dsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       double * beta, double * c, DAAL_INT * ldc)
    {
        openblas_thread_setter ots(1);
        dsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                      const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                       const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xaxpy(DAAL_INT * n, double * a, double * x, DAAL_INT * incx, double * y, DAAL_INT * incy) { daxpy_(n, a, x, incx, y, incy); }

    static void xxaxpy(const DAAL_INT * n, const double * a, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        daxpy_(n, a, x, incx, y, incy);
    }

    static double xxdot(const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        double res = ddot_(n, x, incx, y, incy);
        return res;
    }

    static void xscal(const DAAL_INT * n, const double * a, double * x, const DAAL_INT * incx) { dscal_(n, a, x, incx); }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct OpenBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        ssyrk_(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                       const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        openblas_thread_setter ots(1);
        ssyrk_(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                     const DAAL_INT * lda)
    {
        ssyr_(uplo, n, alpha, x, incx, a, lda);
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                      const DAAL_INT * lda)
    {
        openblas_thread_setter ots(1);
        ssyr_(uplo, n, alpha, x, incx, a, lda);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                      const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                      const DAAL_INT * ldaty)
    {
        sgemm_(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty)
    {
        openblas_thread_setter ots(1);
        sgemm_(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        ssymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       float * beta, float * c, DAAL_INT * ldc)
    {
        openblas_thread_setter ots(1);
        ssymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                      const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                       const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xaxpy(DAAL_INT * n, float * a, float * x, DAAL_INT * incx, float * y, DAAL_INT * incy) { saxpy_(n, a, x, incx, y, incy); }

    static void xxaxpy(const DAAL_INT * n, const float * a, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        saxpy_(n, a, x, incx, y, incy);
    }

    static float xxdot(const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy)
    {
        openblas_thread_setter ots(1);
        float res = sdot_(n, x, incx, y, incy);
        return res;
    }

    static void xscal(const DAAL_INT * n, const float * a, float * x, const DAAL_INT * incx) { sscal_(n, a, x, incx); }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
