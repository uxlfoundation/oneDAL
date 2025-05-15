/* file: service_spblas.h */
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
//  Template wrappers for Sparse BLAS functions.
//--
*/

#ifndef __SERVICE_SPBLAS_H__
#define __SERVICE_SPBLAS_H__

#include "src/externals/config.h"
#include "src/services/service_arrays.h"

typedef unsigned int uint32_t;

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl>
struct SpBlas
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;

    static void xsyrk(char * uplo, char * trans, SizeType * p, SizeType * n, fpType * alpha, fpType * a, SizeType * lda, fpType * beta, fpType * ata,
                      SizeType * ldata)
    {
        _impl<fpType, cpu>::xsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xcsrmultd(const char * transa, const SizeType * m, const SizeType * n, const SizeType * k, fpType * a, SizeType * ja, SizeType * ia,
                          fpType * b, SizeType * jb, SizeType * ib, fpType * c, SizeType * ldc)
    {
        _impl<fpType, cpu>::xcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc);
    }

    static void xcsrmv(const char * transa, const SizeType * m, const SizeType * k, const fpType * alpha, const char * matdescra, const fpType * val,
                       const SizeType * indx, const SizeType * pntrb, const SizeType * pntre, const fpType * x, const fpType * beta, fpType * y)
    {
        _impl<fpType, cpu>::xcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y);
    }

    static void xcsrmm(const char * transa, const SizeType * m, const SizeType * n, const SizeType * k, const fpType * alpha, const char * matdescra,
                       const fpType * val, const SizeType * indx, const SizeType * pntrb, const fpType * b, const SizeType * ldb, const fpType * beta,
                       fpType * c, const SizeType * ldc)
    {
        _impl<fpType, cpu>::xcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, b, ldb, beta, c, ldc);
    }

    static void xxcsrmm(const char * transa, const SizeType * m, const SizeType * n, const SizeType * k, const fpType * alpha, const char * matdescra,
                        const fpType * val, const SizeType * indx, const SizeType * pntrb, const fpType * b, const SizeType * ldb,
                        const fpType * beta, fpType * c, const SizeType * ldc)
    {
        _impl<fpType, cpu>::xxcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, b, ldb, beta, c, ldc);
    }

private:
    static void csr2csc(size_t n, size_t m, const fpType * a, const size_t * col_idx, const size_t * row_start, fpType * csc_a, uint32_t * row_idx,
                        uint32_t * col_start) // O(NNZ) complexity
    {
        const size_t * ptr = row_start;
        const size_t nz    = ptr[n] - ptr[0];

        for (size_t i = 0; i <= m; i++) col_start[i] = 0;

        /* determine column lengths */
        for (size_t i = 0; i < nz; i++)
        {
            col_start[col_idx[i]]++;
        }

        for (size_t i = 0; i < m; i++)
        {
            col_start[i + 1] += col_start[i];
        }

        /* go through the structure once more. Fill in output matrix. */

        for (size_t i = 0; i < n; i++, ptr++)
        {
            for (size_t j = (*ptr) - row_start[0]; j < *(ptr + 1) - row_start[0]; j++)
            {
                size_t k   = col_idx[j] - 1; // we have forced one-based indexing in CSR, but need zero-based in CSC
                size_t l   = col_start[k]++;
                row_idx[l] = i;
                csc_a[l]   = a[j];
            }
        }

        /* shift back col_start */
        for (size_t i = m; i > 0; i--)
        {
            col_start[i] = col_start[i - 1];
        }

        col_start[0] = 0;
    }

    static void splitCSR2CSC(const fpType * a, const size_t * ja, const size_t * ia, size_t n, size_t nRowsInCommonBlock, size_t nRowsInTailBlock,
                             size_t nBlocks, fpType * valuesCSC, uint32_t * colIdxCSC, uint32_t * rowIdxCSC)
    {
        daal::conditional_threader_for((n > 512), nBlocks, [=](size_t i) {
            size_t offset = i * nRowsInCommonBlock;

            uint32_t * rowIdxCSC_i = rowIdxCSC + ia[offset] - ia[0];
            uint32_t * colIdxCSC_i = colIdxCSC + i * (n + 1);
            fpType * valuesCSC_i   = valuesCSC + ia[offset] - ia[0];

            const fpType * a_i  = a + ia[offset] - ia[0];
            const size_t * ja_i = ja + ia[offset] - ia[0];
            const size_t * ia_i = ia + offset;

            size_t nRowsInBlock = nRowsInCommonBlock;
            if (i == nBlocks - 1) nRowsInBlock = nRowsInTailBlock;
            csr2csc(nRowsInBlock, n, a_i, ja_i, ia_i, valuesCSC_i, rowIdxCSC_i, colIdxCSC_i);
        });
    }

    struct CSCBlock
    {
        fpType * values;
        uint32_t * colIdx;
        uint32_t * rowIdx;
    };

    struct DenseBlock
    {
        size_t stride;
        fpType * ptr; // ptr to first element
    };

    static void csc_mm_a_bt(size_t nCols, const CSCBlock & block1, const CSCBlock & block2, DenseBlock & res)
    {
        for (size_t i = 0; i < nCols; ++i)
        {
            const fpType * column1   = block1.values + block1.colIdx[i];        // pointer to column in block1
            const uint32_t nnzCol1   = block1.colIdx[i + 1] - block1.colIdx[i]; // number of non-zero vaules in column1
            const uint32_t * rowPtr1 = block1.rowIdx + block1.colIdx[i];        // indices of non-zero elements in column1

            const fpType * column2   = block2.values + block2.colIdx[i];        // obtain same column from second block
            const uint32_t nnzCol2   = block2.colIdx[i + 1] - block2.colIdx[i]; // and its information ...
            const uint32_t * rowPtr2 = block2.rowIdx + block2.colIdx[i];

            for (size_t ind1 = 0; ind1 < nnzCol1; ++ind1)
            {
                fpType * ptr_ = res.ptr + rowPtr1[ind1] * res.stride;
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t ind2 = 0; ind2 < nnzCol2; ++ind2)
                {
                    ptr_[rowPtr2[ind2]] += column1[ind1] * column2[ind2];
                }
            }
        }
    }

public:
    static services::Status xsyrk_a_at(const fpType * a, const size_t * ja, const size_t * ia, size_t m, size_t n, fpType * c, const size_t ldC)
    {
        size_t nBlocks = 50;
        if (m < nBlocks) nBlocks = 1;

        const size_t nRowsInCommonBlock = m / nBlocks;
        const size_t nRowsInTailBlock   = nRowsInCommonBlock + m % nBlocks;

        const size_t nnzTotal = ia[m] - ia[0];
        const fpType zero     = fpType(0.0);

        TArray<uint32_t, cpu> rowIdxCSCArr(nnzTotal);
        uint32_t * rowIdxCSC = rowIdxCSCArr.get();

        TArray<uint32_t, cpu> colIdxCSCArr((n + 1) * nBlocks);
        uint32_t * colIdxCSC = colIdxCSCArr.get();

        TArray<fpType, cpu> valuesCSCArr(nnzTotal);
        fpType * valuesCSC = valuesCSCArr.get();

        DAAL_CHECK(rowIdxCSC && colIdxCSC && valuesCSC, services::ErrorMemoryAllocationFailed);

        splitCSR2CSC(a, ja, ia, n, nRowsInCommonBlock, nRowsInTailBlock, nBlocks, valuesCSC, colIdxCSC, rowIdxCSC);

        daal::conditional_threader_for(m > 512, nBlocks * nBlocks, [=](size_t idx) {
            const size_t i = idx / nBlocks;
            const size_t j = idx % nBlocks;

            if (i < j) return; // compute only lower traingular part

            CSCBlock block1, block2;
            DenseBlock block_res;
            block_res.stride = ldC;

            SizeType offset_i = i * nRowsInCommonBlock;
            SizeType offset_j = j * nRowsInCommonBlock;

            block1.values = valuesCSC + ia[offset_i] - ia[0];
            block1.colIdx = colIdxCSC + i * (n + 1);
            block1.rowIdx = rowIdxCSC + ia[offset_i] - ia[0];

            block2.values = valuesCSC + ia[offset_j] - ia[0];
            block2.colIdx = colIdxCSC + j * (n + 1);
            block2.rowIdx = rowIdxCSC + ia[offset_j] - ia[0];

            block_res.ptr = c + i * nRowsInCommonBlock * ldC + j * nRowsInCommonBlock;

            size_t rows = nRowsInCommonBlock;
            if (i == nBlocks - 1) rows = nRowsInTailBlock;

            size_t cols = nRowsInCommonBlock;
            if (j == nBlocks - 1) cols = nRowsInTailBlock;

            for (size_t row = 0; row < rows; ++row)
            {
                services::internal::service_memset_seq<fpType, cpu>(&block_res.ptr[row * block_res.stride], zero, cols);
            }

            csc_mm_a_bt(n, block1, block2, block_res);
        });

        return services::Status();
    }

    static services::Status xgemm_a_bt(const fpType * a, const size_t * ja, const size_t * ia, const fpType * b, const size_t * jb, const size_t * ib,
                                       size_t ma, size_t mb, size_t n, fpType * c, const size_t ldC)
    {
        const size_t nRowsInCommonBlock_a = 512;
        const size_t nRowsInCommonBlock_b = 512;

        size_t nBlocks_a = ma / nRowsInCommonBlock_a + !!(ma % nRowsInCommonBlock_a);
        size_t nBlocks_b = mb / nRowsInCommonBlock_b + !!(mb % nRowsInCommonBlock_b);

        const size_t nRowsInTailBlock_a = ma - (nBlocks_a - 1) * nRowsInCommonBlock_a;
        const size_t nRowsInTailBlock_b = mb - (nBlocks_b - 1) * nRowsInCommonBlock_b;

        const fpType zero = fpType(0.0);

        const size_t nnzTotal_a = ia[ma] - ia[0];
        const size_t nnzTotal_b = ib[mb] - ib[0];

        TArray<uint32_t, cpu> rowIdxCSCArr_a(nnzTotal_a);
        uint32_t * rowIdxCSC_a = rowIdxCSCArr_a.get();

        TArray<uint32_t, cpu> colIdxCSCArr_a((n + 1) * nBlocks_a);
        uint32_t * colIdxCSC_a = colIdxCSCArr_a.get();

        TArray<fpType, cpu> valuesCSCArr_a(nnzTotal_a);
        fpType * valuesCSC_a = valuesCSCArr_a.get();

        TArray<uint32_t, cpu> rowIdxCSCArr_b(nnzTotal_b);
        uint32_t * rowIdxCSC_b = rowIdxCSCArr_b.get();

        TArray<uint32_t, cpu> colIdxCSCArr_b((n + 1) * nBlocks_b);
        uint32_t * colIdxCSC_b = colIdxCSCArr_b.get();

        TArray<fpType, cpu> valuesCSCArr_b(nnzTotal_b);
        fpType * valuesCSC_b = valuesCSCArr_b.get();

        DAAL_CHECK(rowIdxCSC_a && colIdxCSC_a && valuesCSC_a && rowIdxCSC_b && colIdxCSC_b && valuesCSC_b, services::ErrorMemoryAllocationFailed);

        splitCSR2CSC(a, ja, ia, n, nRowsInCommonBlock_a, nRowsInTailBlock_a, nBlocks_a, valuesCSC_a, colIdxCSC_a, rowIdxCSC_a);
        splitCSR2CSC(b, jb, ib, n, nRowsInCommonBlock_b, nRowsInTailBlock_b, nBlocks_b, valuesCSC_b, colIdxCSC_b, rowIdxCSC_b);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, ma, mb);

        daal::conditional_threader_for((ma * mb > 512 * 512), nBlocks_a * nBlocks_b, [=](size_t idx) {
            const size_t i = idx / nBlocks_b;
            const size_t j = idx % nBlocks_b;

            CSCBlock block1, block2;
            DenseBlock block_res;
            block_res.stride = ldC;

            const size_t offset_i = i * nRowsInCommonBlock_a;
            const size_t offset_j = j * nRowsInCommonBlock_b;

            block1.values = valuesCSC_a + ia[offset_i] - ia[0];
            block1.colIdx = colIdxCSC_a + i * (n + 1);
            block1.rowIdx = rowIdxCSC_a + ia[offset_i] - ia[0];

            block2.values = valuesCSC_b + ib[offset_j] - ib[0];
            block2.colIdx = colIdxCSC_b + j * (n + 1);
            block2.rowIdx = rowIdxCSC_b + ib[offset_j] - ib[0];

            block_res.ptr = c + i * nRowsInCommonBlock_a * block_res.stride + j * nRowsInCommonBlock_b;

            size_t rows = nRowsInCommonBlock_a;
            if (i == nBlocks_a - 1) rows = nRowsInTailBlock_a;

            size_t cols = nRowsInCommonBlock_b;
            if (j == nBlocks_b - 1) cols = nRowsInTailBlock_b;

            for (size_t row = 0; row < rows; ++row)
            {
                services::internal::service_memset_seq<fpType, cpu>(&block_res.ptr[row * block_res.stride], zero, cols);
            }

            csc_mm_a_bt(n, block1, block2, block_res);
        });

        return services::Status();
    }
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using SpBlasInst = SpBlas<fpType, cpu, SpBlasBackend>;
} // namespace internal
} // namespace daal

#endif
