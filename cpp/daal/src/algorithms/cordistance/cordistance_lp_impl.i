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
services::Status corDistanceLowerPacked(const NumericTable * xTable, const NumericTable * yTable, NumericTable * rTable)
{
    size_t p         = xTable->getNumberOfColumns(); /* Dimension of input feature vector */
    size_t nVectors1 = xTable->getNumberOfRows();    /* Number of input feature vectors in X */
    size_t nVectors2 = yTable->getNumberOfRows();    /* Number of input feature vectors in Y */

    WritePacked<algorithmFPType, cpu> rBlock(rTable);
    DAAL_CHECK_BLOCK_STATUS(rBlock)
    algorithmFPType * r = rBlock.get();

    size_t nBlocks1 = nVectors1 / blockSizeDefault;
    nBlocks1 += (nBlocks1 * blockSizeDefault != nVectors1);

    size_t nBlocks2 = nVectors2 / blockSizeDefault;
    nBlocks2 += (nBlocks2 * blockSizeDefault != nVectors2);

    SafeStatus safeStat;

    /* Process blocks of rows from X and Y */
    daal::threader_for(nBlocks1, nBlocks1, [=, &safeStat](size_t k1) {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks1 - 1)
        {
            blockSize1 = nVectors1 - k1 * blockSizeDefault;
        }

        /* Read blockSize1 rows from X */
        ReadRows<algorithmFPType, cpu> xBlock(*const_cast<NumericTable *>(xTable), k1 * blockSizeDefault, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(xBlock)
        const algorithmFPType * x = xBlock.get();

        /* Compute means for rows in X block */
        algorithmFPType * xMean = (algorithmFPType *)daal::services::daal_malloc(blockSize1 * sizeof(algorithmFPType));
        DAAL_CHECK_MALLOC_THR(xMean)

        for (size_t i = 0; i < blockSize1; i++)
        {
            xMean[i] = 0.0;
            for (size_t j = 0; j < p; j++)
            {
                xMean[i] += x[i * p + j];
            }
            xMean[i] /= p;
        }

        /* For each row in X, compute its squared norm after centering */
        algorithmFPType * xNorm = (algorithmFPType *)daal::services::daal_malloc(blockSize1 * sizeof(algorithmFPType));
        DAAL_CHECK_MALLOC_THR(xNorm)

        for (size_t i = 0; i < blockSize1; i++)
        {
            xNorm[i] = 0.0;
            for (size_t j = 0; j < p; j++)
            {
                algorithmFPType x_centered = x[i * p + j] - xMean[i];
                xNorm[i] += x_centered * x_centered;
            }
            if (xNorm[i] > 0.0)
            {
                xNorm[i] = 1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(xNorm[i]);
            }
        }

        for (size_t k2 = 0; k2 < nBlocks2; k2++)
        {
            DAAL_INT blockSize2 = blockSizeDefault;
            if (k2 == nBlocks2 - 1)
            {
                blockSize2 = nVectors2 - k2 * blockSizeDefault;
            }

            /* Read blockSize2 rows from Y */
            ReadRows<algorithmFPType, cpu> yBlock(*const_cast<NumericTable *>(yTable), k2 * blockSizeDefault, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(yBlock)
            const algorithmFPType * y = yBlock.get();

            /* Compute means for rows in Y block */
            algorithmFPType * yMean = (algorithmFPType *)daal::services::daal_malloc(blockSize2 * sizeof(algorithmFPType));
            DAAL_CHECK_MALLOC_THR(yMean)

            for (size_t j = 0; j < blockSize2; j++)
            {
                yMean[j] = 0.0;
                for (size_t k = 0; k < p; k++)
                {
                    yMean[j] += y[j * p + k];
                }
                yMean[j] /= p;
            }

            /* For each row in Y, compute its squared norm after centering */
            algorithmFPType * yNorm = (algorithmFPType *)daal::services::daal_malloc(blockSize2 * sizeof(algorithmFPType));
            DAAL_CHECK_MALLOC_THR(yNorm)

            for (size_t j = 0; j < blockSize2; j++)
            {
                yNorm[j] = 0.0;
                for (size_t k = 0; k < p; k++)
                {
                    algorithmFPType y_centered = y[j * p + k] - yMean[j];
                    yNorm[j] += y_centered * y_centered;
                }
                if (yNorm[j] > 0.0)
                {
                    yNorm[j] = 1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(yNorm[j]);
                }
            }

            /* Compute correlation distance for each pair of rows from X and Y blocks */
            algorithmFPType rbuf[blockSizeDefault * blockSizeDefault];
            for (size_t i = 0; i < blockSize1; i++)
            {
                for (size_t j = 0; j < blockSize2; j++)
                {
                    algorithmFPType ip = 0.0;
                    for (size_t k = 0; k < p; k++)
                    {
                        algorithmFPType x_centered = x[i * p + k] - xMean[i];
                        algorithmFPType y_centered = y[j * p + k] - yMean[j];
                        ip += x_centered * y_centered;
                    }

                    if (xNorm[i] > 0.0 && yNorm[j] > 0.0)
                    {
                        rbuf[i * blockSize2 + j] = 1.0 - ip * xNorm[i] * yNorm[j];
                    }
                    else
                    {
                        rbuf[i * blockSize2 + j] = 1.0; // Maximum distance when no variance
                    }
                }
            }

            /* Copy results to lower-packed format */
            size_t shift1 = k1 * blockSizeDefault;
            size_t shift2 = k2 * blockSizeDefault;

            for (size_t i = 0; i < blockSize1; i++)
            {
                for (size_t j = 0; j < blockSize2; j++)
                {
                    const size_t row = shift1 + i;
                    const size_t col = shift2 + j;
                    if (row >= col) // Only store in lower triangular part
                    {
                        r[row * (row + 1) / 2 + col] = rbuf[i * blockSize2 + j];
                    }
                }
            }

            daal::services::daal_free(yMean);
            daal::services::daal_free(yNorm);
        }

        daal::services::daal_free(xMean);
        daal::services::daal_free(xNorm);
    });

    /* Set the diagonal of the distance matrix to zeros */
    algorithmFPType zero = (algorithmFPType)0.0;
    daal::threader_for(nVectors1, nVectors1, [=](size_t i) {
        if (i < nVectors2) // Only set diagonal if both matrices have this row
        {
            r[i * (i + 3) / 2] = zero;
        }
    });

    return safeStat.detach();
}

} // namespace internal
} // namespace correlation_distance
} // namespace algorithms
} // namespace daal
