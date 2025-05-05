/* file: zscore_dense_sum_impl.i */
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
//  Implementation of sumDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_SUM_IMPL_I__
#define __ZSCORE_DENSE_SUM_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
Status ZScoreKernel<algorithmFPType, sumDense, cpu>::computeMeanVariance_thr(NumericTable & inputTable, algorithmFPType * resultMean,
                                                                             algorithmFPType * resultVariance,
                                                                             const daal::algorithms::Parameter & parameter)
{
    const size_t nVectors  = inputTable.getNumberOfRows();
    const size_t nFeatures = inputTable.getNumberOfColumns();

    NumericTablePtr sumTable = inputTable.basicStatistics.get(NumericTableIface::sum);
    DAAL_CHECK(sumTable, ErrorPrecomputedSumNotAvailable);

    ReadRows<algorithmFPType, cpu, NumericTable> sumBlock(sumTable.get(), 0, 1);
    const algorithmFPType * sumArray = sumBlock.get();

    algorithmFPType invN   = algorithmFPType(1.0) / algorithmFPType(nVectors);
    algorithmFPType invNm1 = algorithmFPType(1.0) / (algorithmFPType(nVectors) - algorithmFPType(1.0));

    /* Compute means from sums */
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t j = 0; j < nFeatures; j++)
    {
        resultMean[j]     = sumArray[j] * invN;
        resultVariance[j] = 0;
    }

#define _BLOCK_SIZE_ 256

    /* Split rows by blocks, block size cannot be less than nVectors */
    size_t numRowsInBlock = (nVectors > _BLOCK_SIZE_) ? _BLOCK_SIZE_ : nVectors;
    /* Number of blocks */
    size_t numBlocks = nVectors / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + (nVectors - numBlocks * numRowsInBlock);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, sizeof(algorithmFPType));

    /* TLS data initialization */
    daal::tls<algorithmFPType *> tls_data([&]() { return service_scalable_calloc<algorithmFPType, cpu>(nFeatures); });

    SafeStatus safeStat;
    /* Compute partial unscaled variances for each block */
    daal::threader_for(numBlocks, numBlocks, [&](int iBlock) {
        algorithmFPType * pVariances = tls_data.local();
        DAAL_CHECK_THR(pVariances, ErrorMemoryAllocationFailed);

        size_t _nRows    = (iBlock < (numBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
        size_t _startRow = iBlock * numRowsInBlock;

        ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(inputTable, _startRow, _nRows);
        DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
        const algorithmFPType * dataArray_local = dataTableBD.get();

        for (size_t i = 0; i < _nRows; i++)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nFeatures; j++)
            {
                algorithmFPType _v = dataArray_local[i * nFeatures + j] - resultMean[j];
                pVariances[j] += (_v * _v);
            }
        }
    });

    /* Merge unscaled variance arrays by blocks */
    tls_data.reduce([&](algorithmFPType * pVariances) {
        if (pVariances)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nFeatures; j++)
            {
                resultVariance[j] += pVariances[j];
            }
        }
        service_scalable_free<algorithmFPType, cpu>(pVariances);
    });
    /* Convert array of variances to unbiased */

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t j = 0; j < nFeatures; j++)
    {
        resultVariance[j] *= invNm1;
    }

    return safeStat.detach();
}

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
