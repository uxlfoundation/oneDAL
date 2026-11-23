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
#include "services/error_handling.h"
#include <mkl_spblas.h>
#include <mkl_service.h>

namespace daal
{
namespace internal
{
namespace mkl
{
/// MKL's sparse status as a DAAL status: allocation failures keep their meaning, any other failure is internal.
inline services::Status sparseStatus(const sparse_status_t st)
{
    if (st == SPARSE_STATUS_SUCCESS) return services::Status();
    return services::Status(st == SPARSE_STATUS_ALLOC_FAILED ? services::ErrorMemoryAllocationFailed :
                                                               services::ErrorIncorrectInternalFunctionParameter);
}

/// Releases a handle that may not have been created; an earlier failure is kept over the release status.
inline sparse_status_t destroyHandle(sparse_matrix_t handle, const sparse_status_t st)
{
    if (handle == NULL) return st;
    const sparse_status_t destroyed = mkl_sparse_destroy(handle);
    return (st == SPARSE_STATUS_SUCCESS) ? destroyed : st;
}

inline sparse_operation_t sparseOperation(const char * trans)
{
    return (*trans == 'n' || *trans == 'N') ? SPARSE_OPERATION_NON_TRANSPOSE : SPARSE_OPERATION_TRANSPOSE;
}

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

    static services::Status xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, double * a, DAAL_INT * ja,
                                      DAAL_INT * ia, double * b, DAAL_INT * jb, DAAL_INT * ib, double * c, DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        sparse_matrix_t csrB = NULL;
        sparse_status_t st = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*n), (MKL_INT *)ia,
                                                     (MKL_INT *)ia + 1, (MKL_INT *)ja, a);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k), (MKL_INT *)ib,
                                         (MKL_INT *)ib + 1, (MKL_INT *)jb, b);
        }
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_d_spmmd(sparseOperation(transa), csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, static_cast<MKL_INT>(*ldc));
        }
        st = destroyHandle(csrA, st);
        st = destroyHandle(csrB, st);
        return sparseStatus(st);
    }

    static services::Status xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const double * alpha, const char * matdescra,
                                   const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const double * x,
                                   const double * beta, double * y)
    {
        struct matrix_descr descrA;
        descrA.type          = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csrA = NULL;
        sparse_status_t st   = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k),
                                                       (MKL_INT *)pntrb, (MKL_INT *)pntre, (MKL_INT *)indx, (double *)val);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_d_mv(sparseOperation(transa), *alpha, csrA, descrA, x, *beta, y);
        }
        st = destroyHandle(csrA, st);
        return sparseStatus(st);
    }

    static services::Status xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha,
                                    const char * matdescra, const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const double * b,
                                    const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        struct matrix_descr descrA;
        descrA.type          = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csrA = NULL;
        sparse_status_t st   = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k),
                                                       (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1), (MKL_INT *)indx, (double *)val);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_d_mm(sparseOperation(transa), *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, static_cast<MKL_INT>(*n),
                                 static_cast<MKL_INT>(*ldb), *beta, c, static_cast<MKL_INT>(*ldc));
        }
        st = destroyHandle(csrA, st);
        mkl_set_num_threads_local(old_nthr);
        return sparseStatus(st);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklSpBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static services::Status xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, float * a, DAAL_INT * ja,
                                      DAAL_INT * ia, float * b, DAAL_INT * jb, DAAL_INT * ib, float * c, DAAL_INT * ldc)
    {
        sparse_matrix_t csrA = NULL;
        sparse_matrix_t csrB = NULL;
        sparse_status_t st = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*n), (MKL_INT *)ia,
                                                     (MKL_INT *)ia + 1, (MKL_INT *)ja, a);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_s_create_csr(&csrB, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k), (MKL_INT *)ib,
                                         (MKL_INT *)ib + 1, (MKL_INT *)jb, b);
        }
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_s_spmmd(sparseOperation(transa), csrA, csrB, SPARSE_LAYOUT_COLUMN_MAJOR, c, static_cast<MKL_INT>(*ldc));
        }
        st = destroyHandle(csrA, st);
        st = destroyHandle(csrB, st);
        return sparseStatus(st);
    }

    static services::Status xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const float * alpha, const char * matdescra,
                                   const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const float * x,
                                   const float * beta, float * y)
    {
        struct matrix_descr descrA;
        descrA.type          = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csrA = NULL;
        sparse_status_t st   = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k),
                                                       (MKL_INT *)pntrb, (MKL_INT *)pntre, (MKL_INT *)indx, (float *)val);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_s_mv(sparseOperation(transa), *alpha, csrA, descrA, x, *beta, y);
        }
        st = destroyHandle(csrA, st);
        return sparseStatus(st);
    }

    static services::Status xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha,
                                    const char * matdescra, const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const float * b,
                                    const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        struct matrix_descr descrA;
        descrA.type          = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t csrA = NULL;
        sparse_status_t st   = mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, static_cast<MKL_INT>(*m), static_cast<MKL_INT>(*k),
                                                       (MKL_INT *)pntrb, (MKL_INT *)(pntrb + 1), (MKL_INT *)indx, (float *)val);
        if (st == SPARSE_STATUS_SUCCESS)
        {
            st = mkl_sparse_s_mm(sparseOperation(transa), *alpha, csrA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, b, static_cast<MKL_INT>(*n),
                                 static_cast<MKL_INT>(*ldb), *beta, c, static_cast<MKL_INT>(*ldc));
        }
        st = destroyHandle(csrA, st);
        mkl_set_num_threads_local(old_nthr);
        return sparseStatus(st);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
