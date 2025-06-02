/* file: service_spblas_mkl.h */
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

#ifndef __SERVICE_SPBLAS_MKL_H__
#define __SERVICE_SPBLAS_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklSpBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklSpBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, double * a, DAAL_INT * ja, DAAL_INT * ia,
                          double * b, DAAL_INT * jb, DAAL_INT * ib, double * c, DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*n, (MKL_INT *)ia, (MKL_INT *)ia + 1, (MKL_INT *)ja,
                                a);

        sparse_matrix_t csrB = NULL;
        struct matrix_descr descrB;
        descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)ib, (MKL_INT *)ib + 1, (MKL_INT *)jb,
                                b);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_d_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_d_spmmd(SPARSE_OPERATION_TRANSPOSE, csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);
        mkl_sparse_destroy(csrB);
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const double * alpha, const char * matdescra, const double * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const double * x, const double * beta, double * y)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)pntre,
                                (MKL_INT *)indx, (double *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, x, *beta, y);
        }
        else
        {
            mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, x, *beta, y);
        }
        mkl_sparse_destroy(csrA);
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const char * matdescra,
                       const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const double * b, const DAAL_INT * ldb, const double * beta,
                       double * c, const DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1),
                                (MKL_INT *)indx, (double *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n,
                            (const MKL_INT)*ldb, *beta, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n, (const MKL_INT)*ldb,
                            *beta, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const char * matdescra,
                        const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const double * b, const DAAL_INT * ldb,
                        const double * beta, double * c, const DAAL_INT * ldc)
    {
        int old_nthr         = mkl_set_num_threads_local(1);
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1),
                                (MKL_INT *)indx, (double *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n,
                            (const MKL_INT)*ldb, *beta, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n, (const MKL_INT)*ldb,
                            *beta, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);

        mkl_set_num_threads_local(old_nthr);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklSpBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, float * a, DAAL_INT * ja, DAAL_INT * ia,
                          float * b, DAAL_INT * jb, DAAL_INT * ib, float * c, DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*n, (MKL_INT *)ia, (MKL_INT *)ia + 1, (MKL_INT *)ja,
                                a);

        sparse_matrix_t csrB = NULL;
        struct matrix_descr descrB;
        descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_create_csr(&csrB, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)ib, (MKL_INT *)ib + 1, (MKL_INT *)jb,
                                b);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_s_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_s_spmmd(SPARSE_OPERATION_TRANSPOSE, csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);
        mkl_sparse_destroy(csrB);
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const float * alpha, const char * matdescra, const float * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const float * x, const float * beta, float * y)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)pntre,
                                (MKL_INT *)indx, (float *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, x, *beta, y);
        }
        else
        {
            mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, x, *beta, y);
        }
        mkl_sparse_destroy(csrA);
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const char * matdescra,
                       const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const float * b, const DAAL_INT * ldb, const float * beta,
                       float * c, const DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1),
                                (MKL_INT *)indx, (float *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n,
                            (const MKL_INT)*ldb, *beta, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_s_mm(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n, (const MKL_INT)*ldb,
                            *beta, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const char * matdescra,
                        const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const float * b, const DAAL_INT * ldb, const float * beta,
                        float * c, const DAAL_INT * ldc)
    {
        int old_nthr         = mkl_set_num_threads_local(1);
        sparse_matrix_t csrA = NULL;
        struct matrix_descr descrA;
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, (const MKL_INT)*m, (const MKL_INT)*k, (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1),
                                (MKL_INT *)indx, (float *)val);

        if (*transa == 'n' || *transa == 'N')
        {
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n,
                            (const MKL_INT)*ldb, *beta, c, (const MKL_INT)*ldc);
        }
        else
        {
            mkl_sparse_s_mm(SPARSE_OPERATION_TRANSPOSE, *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, (const MKL_INT)*n, (const MKL_INT)*ldb,
                            *beta, c, (const MKL_INT)*ldc);
        }
        mkl_sparse_destroy(csrA);

        mkl_set_num_threads_local(old_nthr);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
