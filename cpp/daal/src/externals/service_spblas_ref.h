/* file: service_spblas_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//++
//  Template wrappers for common Sparse BLAS functions.
//--
*/

#ifndef __SERVICE_SPBLAS_REF_H__
#define __SERVICE_SPBLAS_REF_H__

#include "src/externals/service_memory.h" // required for memset

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct RefSpBlas
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, fpType * a, DAAL_INT * ja, DAAL_INT * ia,
                          fpType * b, DAAL_INT * jb, DAAL_INT * ib, fpType * c, DAAL_INT * ldc)
    {
        if (*transa == 'n' || *transa == 'N')
        {
            csrmultd(m, n, k, a, ja, ia, b, jb, ib, c, ldc);
        }
        else
        {
            csrmultd_transpose(m, n, k, a, ja, ia, b, jb, ib, c, ldc);
        }
    }

    static void csrmultd(const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, fpType * a, DAAL_INT * ja, DAAL_INT * ia, fpType * b,
                         DAAL_INT * jb, DAAL_INT * ib, fpType * c, DAAL_INT * ldc)
    {
        DAAL_INT indexing = 1; // 1-based indexing
        DAAL_INT row_b, row_c, col_c, val_ptr_a, val_ptr_b;
        fpType a_elt, b_elt;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, *ldc, (*k) - 1);
        for (DAAL_INT col_c = 0; col_c < *k; col_c++) //flush the matrix c
        {
            services::internal::service_memset<fpType, cpu>(c + col_c * (*ldc), fpType(0), *m);
        }
        for (row_c = 0; row_c < *m; row_c++) // row_a = row_c
        {
            for (val_ptr_a = ia[row_c] - indexing; val_ptr_a < ia[row_c + 1] - indexing; val_ptr_a++)
            {
                row_b = ja[val_ptr_a] - indexing;
                a_elt = a[val_ptr_a];
                for (val_ptr_b = ib[row_b] - indexing; val_ptr_b < ib[row_b + 1] - indexing; val_ptr_b++)
                {
                    col_c = jb[val_ptr_b] - indexing;
                    b_elt = b[val_ptr_b];
                    c[col_c * (*ldc) + row_c] += a_elt * b_elt;
                }
            }
        }
    }

    static void csrmultd_transpose(const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, fpType * a, DAAL_INT * ja, DAAL_INT * ia, fpType * b,
                                   DAAL_INT * jb, DAAL_INT * ib, fpType * c, DAAL_INT * ldc)
    {
        DAAL_INT indexing = 1;
        DAAL_INT row_a, row_b, row_c, col_c, val_ptr_a, val_ptr_b;
        fpType a_elt, b_elt;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, *ldc, (*n) - 1);
        for (DAAL_INT col_c = 0; col_c < *k; col_c++) //flush the matrix c
        {
            services::internal::service_memset<fpType, cpu>(c + col_c * (*ldc), fpType(0), *n);
        }
        for (row_a = 0; row_a < *m; row_a++)
        {
            row_b = row_a;
            for (val_ptr_b = ib[row_b] - indexing; val_ptr_b < ib[row_b + 1] - indexing; val_ptr_b++)
            {
                b_elt = b[val_ptr_b];
                col_c = jb[val_ptr_b] - indexing; //col_c = col_b
                for (val_ptr_a = ia[row_a] - indexing; val_ptr_a < ia[row_a + 1] - indexing; val_ptr_a++)
                {
                    row_c = ja[val_ptr_a] - indexing; //row_c = col_a
                    a_elt = a[val_ptr_a];
                    c[col_c * (*ldc) + row_c] += a_elt * b_elt;
                }
            }
        }
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const fpType * alpha, const char * matdescra, const fpType * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const fpType * x, const fpType * beta, fpType * y)
    {
        if (*transa == 'n' || *transa == 'N')
        {
            csrmv(m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y);
        }
        else
        {
            csrmv_transpose(m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y);
        }
    }

    static void csrmv(const DAAL_INT * m, const DAAL_INT * k, const fpType * alpha, const char * matdescra, const fpType * val, const DAAL_INT * indx,
                      const DAAL_INT * pntrb, const DAAL_INT * pntre, const fpType * x, const fpType * beta, fpType * y)
    {
        DAAL_INT indexing = 1;
        if (matdescra[3] == 'C') indexing = 0; // if fourth entry is 'C' zero based
        DAAL_INT curr_row_start, curr_row_end, i, k_ind;
        for (DAAL_INT row_num = 0; row_num < *m; row_num++)
        {
            y[row_num] *= (*beta);
            curr_row_start = pntrb[row_num] - indexing;
            curr_row_end   = pntre[row_num] - indexing;
            for (i = curr_row_start; i < curr_row_end; i++)
            {
                k_ind = indx[i] - indexing;
                y[row_num] += (*alpha) * x[k_ind] * val[i];
            }
        }
    }

    static void csrmv_transpose(const DAAL_INT * m, const DAAL_INT * k, const fpType * alpha, const char * matdescra, const fpType * val,
                                const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const fpType * x, const fpType * beta,
                                fpType * y)
    {
        DAAL_INT indexing = 1;
        if (matdescra[3] == 'C') indexing = 0; // if fourth entry is 'C' zero based
        for (DAAL_INT _i = 0; _i < *k; _i++)
        {
            y[_i] *= *beta;
        }
        fpType coeff;
        DAAL_INT row_num, i, curr_row_start, curr_row_end;
        for (row_num = 0; row_num < *m; row_num++)
        {
            coeff          = (*alpha) * x[row_num];
            curr_row_start = pntrb[row_num] - indexing;
            curr_row_end   = pntre[row_num] - indexing;
            for (i = curr_row_start; i < curr_row_end; i++)
            {
                y[indx[i] - indexing] += coeff * val[i];
            }
        }
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const char * matdescra,
                       const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb, const fpType * beta,
                       fpType * c, const DAAL_INT * ldc)
    {
        csrmm(m, n, k, alpha, val, indx, pntrb, b, ldb, beta, c, ldc);
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const char * matdescra,
                        const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb,
                        const fpType * beta, fpType * c, const DAAL_INT * ldc)
    {
        csrmm(m, n, k, alpha, val, indx, pntrb, b, ldb, beta, c, ldc);
    }
    inline static fpType csracc(DAAL_INT row, DAAL_INT col, const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb)
    {
        DAAL_INT offset = pntrb[row] - 1;
        DAAL_INT nnz    = pntrb[row + 1] - pntrb[row];
        DAAL_INT csrcol = col + 1;

        for (DAAL_INT i = 0; i < nnz; ++i)
        {
            if (csrcol < indx[offset + i]) break;
            if (csrcol == indx[offset + i]) return val[offset + i];
        }

        return fpType(0);
    }
    static void csrmm(const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const fpType * a, const DAAL_INT * indx,
                      const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb, const fpType * beta, fpType * c, const DAAL_INT * ldc)
    {
        DAAL_INT ldbVal = *ldb;
        DAAL_INT ldcVal = *ldc;
        DAAL_INT nVal   = *n;
        DAAL_INT mVal   = *m;
        DAAL_INT kVal   = *k;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, ldbVal, nVal);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, ldcVal, nVal);
        DAAL_OVERFLOW_CHECK_BY_ADDING_THROW_IF_POSSIBLE(DAAL_INT, ldbVal * ((*n) - 1), kVal);
        DAAL_OVERFLOW_CHECK_BY_ADDING_THROW_IF_POSSIBLE(DAAL_INT, ldcVal * ((*n) - 1), mVal);

        for (DAAL_INT mInd = 0; mInd < mVal; ++mInd)
            for (DAAL_INT nInd = 0; nInd < *n; ++nInd)
            {
                fpType sum = 0.0;
                for (DAAL_INT kInd = 0; kInd < kVal; ++kInd)
                {
                    fpType ail  = csracc(mInd, kInd, a, indx, pntrb);
                    fpType btlj = b[ldbVal * nInd + kInd];
                    sum += ail * btlj;
                }
                c[nInd * ldcVal + mInd] *= *beta;
                c[nInd * ldcVal + mInd] += *alpha * sum;
            }
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
