/* file: cordistance_full_impl.i */
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
//  Implementation of correlation distance for result in full layout.
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
services::Status corDistanceFull(const NumericTable * xTable, const NumericTable * yTable, NumericTable * rTable)
{
    size_t p = xTable->getNumberOfColumns(); /* Dimension of input feature vector */
    size_t nVectors1 = xTable->getNumberOfRows();    /* Number of input vectors in X */
    size_t nVectors2 = yTable->getNumberOfRows();    /* Number of input vectors in Y */

    size_t nBlocks1 = nVectors1 / blockSizeDefault;
    nBlocks1 += (nBlocks1 * blockSizeDefault != nVectors1);

    SafeStatus safeStat;

    /* compute results for blocks of the distance matrix */
    daal::threader_for(nBlocks1, nBlocks1, [=, &safeStat](size_t k1) {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks1 - 1)
        {
            blockSize1 = nVectors1 - k1 * blockSizeDefault;
        }

        /* read access to blockSize1 rows in input dataset X at k1*blockSizeDefault*p row */
        ReadRows<algorithmFPType, cpu> xBlock(*const_cast<NumericTable *>(xTable), k1 * blockSizeDefault, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(xBlock)
        const algorithmFPType * x = xBlock.get();

        /* write access to blockSize1 rows in output dataset */
        WriteOnlyRows<algorithmFPType, cpu> rBlock(rTable, k1 * blockSizeDefault, blockSize1);
        DAAL_CHECK_BLOCK_STATUS_THR(rBlock)
        algorithmFPType * r = rBlock.get();

        /* Compute means for rows in X block */
        algorithmFPType* xMean = (algorithmFPType*)daal::services::daal_malloc(blockSize1 * sizeof(algorithmFPType));
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

        size_t nBlocks2 = nVectors2 / blockSizeDefault;
        nBlocks2 += (nBlocks2 * blockSizeDefault != nVectors2);

        for (size_t k2 = 0; k2 < nBlocks2; k2++)
        {
            DAAL_INT blockSize2 = blockSizeDefault;
            if (k2 == nBlocks2 - 1)
            {
                blockSize2 = nVectors2 - k2 * blockSizeDefault;
            }

            size_t shift2 = k2 * blockSizeDefault;

            /* read access to blockSize2 rows in input dataset Y */
            ReadRows<algorithmFPType, cpu> yBlock(*const_cast<NumericTable *>(yTable), shift2, blockSize2);
            DAAL_CHECK_BLOCK_STATUS_THR(yBlock)
            const algorithmFPType * y = yBlock.get();

            /* Compute means for rows in Y block */
            algorithmFPType* yMean = (algorithmFPType*)daal::services::daal_malloc(blockSize2 * sizeof(algorithmFPType));
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

            for (size_t i = 0; i < blockSize1; i++)
            {
                for (size_t j = 0; j < blockSize2; j++)
                {
                    algorithmFPType numerator = 0.0;
                    algorithmFPType xNorm = 0.0;
                    algorithmFPType yNorm = 0.0;

                    for (size_t k = 0; k < p; k++)
                    {
                        algorithmFPType x_centered = x[i * p + k] - xMean[i];
                        algorithmFPType y_centered = y[j * p + k] - yMean[j];
                        
                        numerator += x_centered * y_centered;
                        xNorm += x_centered * x_centered;
                        yNorm += y_centered * y_centered;
                    }

                    algorithmFPType denominator = xNorm * yNorm;
                    if (denominator > 0.0)
                    {
                        r[i * nVectors2 + shift2 + j] = 1.0 - numerator / 
                            (daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(xNorm) * 
                             daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(yNorm));
                    }
                    else
                    {
                        r[i * nVectors2 + shift2 + j] = 1.0; // Maximum distance when no variance
                    }
                }
            }
            
            daal::services::daal_free(yMean);
        }
        
        daal::services::daal_free(xMean);
    });

    return safeStat.detach();
}

} // namespace internal
} // namespace correlation_distance
} // namespace algorithms
} // namespace daal
