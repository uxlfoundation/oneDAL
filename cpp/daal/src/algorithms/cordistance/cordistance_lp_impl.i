/* file: cordistance_lp_impl.i */
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
//  Implementation of correlation distance for result in lower triangular layout.
//--
*/
#include "src/services/service_defines.h"
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status corDistanceLowerPacked(const NumericTable * xTable, NumericTable * rTable)
{
    size_t p = xTable->getNumberOfColumns(); /* Dimension of input feature vector */
    size_t n = xTable->getNumberOfRows();    /* Number of input feature vectors   */

    WritePacked<algorithmFPType, cpu> rBlock(rTable);
    DAAL_CHECK_BLOCK_STATUS(rBlock)
    algorithmFPType * r = rBlock.get();

    size_t nBlocks = n / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != n);

    SafeStatus safeStat;

    /* compute major diagonal blocks of the distance matrix */
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](size_t k1) {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks - 1)
        {
            blockSize1 = n - k1 * blockSizeDefault;
        }

        /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault*p row */
        ReadRows<algorithmFPType, cpu> xBlock(*const_cast<NumericTable *>(xTable), k1 * blockSizeDefault, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(xBlock)
        const algorithmFPType * x = xBlock.get();

        algorithmFPType sum[blockSizeDefault], buf[blockSizeDefault * blockSizeDefault];

        /* compute sums for elements in each row of the block */
        for (size_t i = 0; i < blockSize1; i++)
        {
            algorithmFPType s = (algorithmFPType)0.0;
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < p; j++)
            {
                s += x[i * p + j];
            }
            sum[i] = s;
        }

        /* calculate sum^t * sum */
        const algorithmFPType one(1.0);
        const algorithmFPType zero(0.0);
        algorithmFPType alpha = one, beta = zero;
        char transa = 'N', transb = 'T';
        DAAL_INT m = blockSize1, k = 1, nn = blockSize1;
        DAAL_INT lda = m, ldb = nn, ldc = m;

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, sum, &lda, sum, &ldb, &beta, buf, &ldc);

        /* calculate x * x^t - 1/p * sum^t * sum */
        alpha  = one;
        beta   = -one / (algorithmFPType)p;
        transa = 'T';
        transb = 'N';
        m = blockSize1, k = p, nn = blockSize1;
        lda = k;
        ldb = k;
        ldc = m;

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x, &lda, x, &ldb, &beta, buf, &ldc);

        /* compute inverse of sqrt of gemm result and save for use in computation off-diagonal blocks */
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < blockSize1; i++)
        {
            if (buf[i * blockSize1 + i] > (algorithmFPType)0.0)
            {
                buf[i * blockSize1 + i] = (algorithmFPType)1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(buf[i * blockSize1 + i]);
            }
        }

        /* compute cosine distance for k1 block of rows in the input dataset */
        for (size_t i = 0; i < blockSize1; i++)
        {
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < i; j++)
            {
                buf[i * blockSize1 + j] = 1.0 - buf[i * blockSize1 + j] * buf[i * blockSize1 + i] * buf[j * blockSize1 + j];
            }
        }

        /* unpack the results into user's memory */
        size_t shift1 = k1 * blockSizeDefault;
        /* beginning of the memory to copy in starts at shift1 * ( shift1 + 1 ) / 2 + shift1 position */
        algorithmFPType * rr = r + shift1 * (shift1 + 1) / 2 + shift1;
        for (size_t i = 0; i < blockSize1; i++)
        {
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j <= i; j++)
            {
                rr[j] = buf[i * blockSize1 + j];
            }
            /* the next "row" in the user memory is shifted by shift1+i+1 positions vs the previous one */
            rr += (shift1 + i + 1);
        }
    });
    DAAL_CHECK_SAFE_STATUS()

    /* compute off-diagonal blocks of the distance matrix */
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](size_t k1) {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks - 1)
        {
            blockSize1 = n - k1 * blockSizeDefault;
        }

        size_t shift1 = k1 * blockSizeDefault;

        /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault row */
        ReadRows<algorithmFPType, cpu> xBlock1(*const_cast<NumericTable *>(xTable), shift1, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(xBlock1)
        const algorithmFPType * x1 = xBlock1.get();

        algorithmFPType sum1[blockSizeDefault];

        /* compute sums for elements in each row of the block x1 */
        for (size_t i = 0; i < blockSize1; i++)
        {
            algorithmFPType s = (algorithmFPType)0.0;
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < p; j++)
            {
                s += x1[i * p + j];
            }
            sum1[i] = s;
        }

        /* compute upper triangular of the distance matrix */
        daal::threader_for(nBlocks - k1 - 1, nBlocks - k1 - 1, [=, &safeStat, &sum1](size_t k3) {
            DAAL_INT blockSize2 = blockSizeDefault;
            size_t k2           = k3 + k1 + 1;
            size_t nl = n, pl = p;
            algorithmFPType * sum1l = const_cast<algorithmFPType *>(sum1);

            if (k2 == nBlocks - 1)
            {
                blockSize2 = nl - k2 * blockSizeDefault;
            }

            /* extract diagonal elements of k1-th diagonal block of the matrix */
            size_t shift1l = shift1, idx;
            algorithmFPType *rr, diag1[blockSizeDefault], diag2[blockSizeDefault], buf[blockSizeDefault * blockSizeDefault];

            /* shift to the last diagonal element of the k1-1 block */
            rr  = r + (shift1l * (shift1l + 1) / 2 - 1);
            idx = 0;
            for (size_t i = 0; i < blockSize1; i++)
            {
                /* the next diagonal element is shift1+1+i elements from the current in the packed array */
                idx += (shift1l + i + 1);
                diag1[i] = rr[idx];
            }

            /* extract diagonal elements of k2-th diagonal block of the matrix */
            /* shift to the last diagonal element of the k2-1 block */
            size_t shift2 = k2 * blockSizeDefault, shift2l = shift2;
            rr  = r + (shift2l * (shift2l + 1) / 2 - 1);
            idx = 0;
            for (size_t i = 0; i < blockSize2; i++)
            {
                /* the next diagonal element is shift2+1+i elements from the current in the packed array */
                idx += (shift2l + i + 1);
                diag2[i] = rr[idx];
            }

            /* read access to blockSize2 rows in input dataset at k2*blockSizeDefault row */
            ReadRows<algorithmFPType, cpu> xBlock2(*const_cast<NumericTable *>(xTable), shift2, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(xBlock2)
            const algorithmFPType * x2 = xBlock2.get();

            algorithmFPType sum2[blockSizeDefault];

            /* compute sums for elements in each row of the block x2 */
            for (size_t i = 0; i < blockSize2; i++)
            {
                algorithmFPType s = (algorithmFPType)0.0;
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < pl; j++)
                {
                    s += x2[i * pl + j];
                }
                sum2[i] = s;
            }

            /* calculate sum1^t * sum2 */
            const algorithmFPType one(1.0);
            const algorithmFPType zero(0.0);
            algorithmFPType alpha = 1.0, beta = zero;
            char transa = 'N', transb = 'T';
            DAAL_INT m = blockSize2, k = 1, nn = blockSize1;
            DAAL_INT lda = m, ldb = nn, ldc = m;

            BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, sum2, &lda, sum1l, &ldb, &beta, buf, &ldc);

            /* calculate x1 * x2^t - 1/p * sum1^t * sum2 */
            alpha  = one;
            beta   = -one / (algorithmFPType)pl;
            transa = 'T';
            transb = 'N';
            m      = blockSize2;
            k      = pl;
            nn     = blockSize1;
            lda    = k;
            ldb    = k;
            ldc    = m;

            /* compute the distance between k1 and k2 blocks of rows in the input dataset */
            BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x2, &lda, x1, &ldb, &beta, buf, &ldc);

            for (size_t i = 0; i < blockSize1; i++)
            {
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < blockSize2; j++)
                {
                    buf[i * blockSize2 + j] = 1.0 - buf[i * blockSize2 + j] * diag1[i] * diag2[j];
                }
            }

            /* copy the results into user memory */
            /* The memory to copy in starts at the position  shift2 * ( shift2 + 1 ) / 2 + shift1 */
            rr = r + shift2l * (shift2l + 1) / 2 + shift1l;
            for (size_t i = 0; i < blockSize2; i++)
            {
                for (size_t j = 0; j < blockSize1; j++)
                {
                    rr[j] = buf[j * blockSize2 + i];
                }

                /* the next "row" of the distance is shifted by shift2 + i + 1 elements vs previous */
                rr += (shift2l + i + 1);
            }
        });
        if (!safeStat)
        {
            return;
        }
    });
    DAAL_CHECK_SAFE_STATUS()

    /* set the diagonal of the distance matrix to zeros */
    algorithmFPType zero = (algorithmFPType)0.0;
    daal::threader_for(n, n, [=](size_t i) { r[i * (i + 3) / 2] = zero; });

    return safeStat.detach();
}

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
