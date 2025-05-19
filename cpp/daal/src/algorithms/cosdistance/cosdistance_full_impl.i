/* file: cosdistance_full_impl.i */
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
//  Implementation of cosine distance for result in full layout.
//--
*/
#include "src/services/service_defines.h"
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status cosDistanceFull(const NumericTable * xTable, NumericTable * rTable)
{
    size_t p = xTable->getNumberOfColumns(); /* Dimension of input feature vector */
    size_t n = xTable->getNumberOfRows();    /* Number of input feature vectors   */

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

        /* write access to blockSize1 rows in output dataset at k1*blockSizeDefault*p row */
        WriteOnlyRows<algorithmFPType, cpu> rBlock(rTable, k1 * blockSizeDefault, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(rBlock)
        algorithmFPType * r = rBlock.get();

        /* move to respective column position in output dataset */
        algorithmFPType * rr = r + k1 * blockSizeDefault;

        algorithmFPType buf[blockSizeDefault * blockSizeDefault];
        algorithmFPType bufDiag[blockSizeDefault];
        algorithmFPType alpha = 1.0, beta = 0.0;
        char transa = 'T', transb = 'N';
        DAAL_INT m = blockSize1, k = p, nn = blockSize1;
        DAAL_INT lda = k, ldb = p, ldc = m;

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x, &lda, x, &ldb, &beta, buf, &ldc);

        for (size_t i = 0; i < blockSize1; i++)
        {
            if (buf[i * blockSize1 + i] > (algorithmFPType)0.0)
            {
                buf[i * blockSize1 + i] = (algorithmFPType)1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(buf[i * blockSize1 + i]);
            }
            bufDiag[i] = buf[i * blockSize1 + i];
        }

        for (size_t i = 0; i < blockSize1; i++)
        {
            // PRAGMA_FORCE_SIMD
            // PRAGMA_VECTOR_ALWAYS
            for (size_t j = i + 1; j < blockSize1; j++)
            {
                buf[i * blockSize1 + j] = 1.0 - buf[i * blockSize1 + j] * bufDiag[i] * bufDiag[j];
            }
        }

        for (size_t i = 0; i < blockSize1; i++)
        {
            // PRAGMA_VECTOR_ALWAYS
            for (size_t j = i; j < blockSize1; j++)
            {
                rr[i * n + j] = buf[i * blockSize1 + j];
            }
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

        daal::threader_for(nBlocks - k1 - 1, nBlocks - k1 - 1, [=, &safeStat](size_t k3) {
            DAAL_INT blockSize2 = blockSizeDefault;
            size_t k2           = k3 + k1 + 1;
            size_t nl           = n;

            if (k2 == nBlocks - 1)
            {
                blockSize2 = nl - k2 * blockSizeDefault;
            }

            size_t shift2 = k2 * blockSizeDefault;

            /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault row */
            ReadRows<algorithmFPType, cpu> xBlock2(*const_cast<NumericTable *>(xTable), shift2, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(xBlock2)
            const algorithmFPType * x2 = xBlock2.get();

            algorithmFPType diag[blockSizeDefault], buf[blockSizeDefault * blockSizeDefault];

            WriteOnlyRows<algorithmFPType, cpu> rBlock(rTable, shift2, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(rBlock)
            algorithmFPType * r = rBlock.get();

            for (size_t i = 0; i < blockSize2; i++)
            {
                diag[i] = r[i * nl + shift2 + i];
            }

            /* write access to blockSize2 rows in output dataset at k2*blockSizeDefault row */
            r = rBlock.next(shift1, blockSize1);
            DAAL_CHECK_BLOCK_STATUS_THR(rBlock)
            /* move to respective column position in output dataset */
            algorithmFPType * rr = r + shift2;

            algorithmFPType alpha = 1.0, beta = 0.0;
            char transa = 'T', transb = 'N';
            DAAL_INT m = blockSize2, k = p, nn = blockSize1;
            DAAL_INT lda = k, ldb = p, ldc = m;

            BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x2, &lda, x1, &ldb, &beta, buf, &ldc);

            for (size_t i = 0; i < blockSize1; i++)
            {
                // PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < blockSize2; j++)
                {
                    buf[i * blockSize2 + j] = 1.0 - buf[i * blockSize2 + j] * r[i * nl + (shift1 + i)] * diag[j];
                }
            }

            for (size_t i = 0; i < blockSize1; i++)
            {
                // PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < blockSize2; j++)
                {
                    rr[i * nl + j] = buf[i * blockSize2 + j];
                }
            }
        });
    });
    DAAL_CHECK_SAFE_STATUS()

    // copy upper triangular of r into lower triangular and unit diagonal
    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](size_t k1) {
        size_t blockSize1 = blockSizeDefault;
        if (k1 == nBlocks - 1)
        {
            blockSize1 = n - k1 * blockSizeDefault;
        }

        size_t shift1 = k1 * blockSizeDefault;

        WriteRows<algorithmFPType, cpu> rBlock1(rTable, shift1, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(rBlock1)
        algorithmFPType * r1 = rBlock1.get();

        algorithmFPType * rr1 = r1 + shift1;

        for (size_t i = 0; i < blockSize1; i++)
        {
            rr1[i * n + i] = 0.0;
            for (size_t j = i + 1; j < blockSize1; j++)
            {
                rr1[j * n + i] = rr1[i * n + j];
            }
        }

        daal::threader_for(nBlocks - k1 - 1, nBlocks - k1 - 1, [=, &safeStat](size_t k3) {
            size_t k2 = k3 + k1 + 1;
            size_t nl = n;
            algorithmFPType *rr1, *rr2;
            size_t blockSize2 = blockSizeDefault;
            if (k2 == nBlocks - 1)
            {
                blockSize2 = nl - k2 * blockSizeDefault;
            }

            size_t shift2 = k2 * blockSizeDefault;

            WriteRows<algorithmFPType, cpu> rBlock2(rTable, shift2, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(rBlock2)
            algorithmFPType * r2 = rBlock2.get();

            rr1 = r1 + shift2;
            rr2 = r2 + shift1;

            for (size_t i = 0; i < blockSize1; i++)
            {
                for (size_t j = 0; j < blockSize2; j++)
                {
                    rr2[j * nl + i] = rr1[i * nl + j];
                }
            }
        });
    });

    return safeStat.detach();
}

} // namespace internal

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal
