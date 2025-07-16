/* file: service_blas_mkl.h */
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

#ifndef __SERVICE_BLAS_MKL_H__
#define __SERVICE_BLAS_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>

#define __DAAL_MKLFN_CALL_BLAS(f_name, f_args) f_name f_args;

#define __DAAL_MKLFN_CALL_RETURN_BLAS(f_name, f_args, res) res = f_name f_args;

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                       const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                      const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                       const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                       const DAAL_INT * ldaty)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       double * beta, double * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                      const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                       const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, double * a, double * x, DAAL_INT * incx, double * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const double * a, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static double xxdot(const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        double res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(ddot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        mkl_set_num_threads_local(old_nthr);
        return res;
    }

    static void xscal(const DAAL_INT * n, const double * a, double * x, const DAAL_INT * incx)
    {
        __DAAL_MKLFN_CALL_BLAS(dscal, ((const MKL_INT *)n, a, x, (const MKL_INT *)incx));
    }

    static double xasum(const DAAL_INT * n, const double * x, const DAAL_INT * incx)
    {
        double res;
        __DAAL_MKLFN_CALL_BLAS(dasum, ((const MKL_INT *)n, x, (const MKL_INT *)incx), res);
        return res;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                       const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                      const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       float * beta, float * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                      const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                       const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, float * a, float * x, DAAL_INT * incx, float * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const float * a, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static float xxdot(const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        float res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(sdot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        mkl_set_num_threads_local(old_nthr);
        return res;
    }

    static void xscal(const DAAL_INT * n, const float * a, float * x, const DAAL_INT * incx)
    {
        __DAAL_MKLFN_CALL_BLAS(sscal, ((const MKL_INT *)n, a, x, (const MKL_INT *)incx));
    }

    static float xasum(const DAAL_INT * n, const float * x, const DAAL_INT * incx)
    {
        float res;
        __DAAL_MKLFN_CALL_BLAS(sasum, ((const MKL_INT *)n, x, (const MKL_INT *)incx), res);
        return res;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
