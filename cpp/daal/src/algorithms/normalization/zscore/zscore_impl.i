/* file: zscore_impl.i */
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

//++
//  Implementation of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_IMPL_I__
#define __ZSCORE_IMPL_I__

#include "src/algorithms/normalization/zscore/zscore_base.h"

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
Status ZScoreKernelBase<algorithmFPType, cpu>::common_compute(NumericTable & inputTable, NumericTable & resultTable, algorithmFPType * mean_total,
                                                              algorithmFPType * variances_total, const daal::algorithms::Parameter & parameter)
{
#define _BLOCK_SIZE_NORM_ 256

    const size_t _nVectors  = inputTable.getNumberOfRows();
    const size_t _nFeatures = inputTable.getNumberOfColumns();

    /* Split rows by blocks, block size cannot be less than _nVectors */
    size_t numRowsInBlock = (_nVectors > _BLOCK_SIZE_NORM_) ? _BLOCK_SIZE_NORM_ : _nVectors;
    /* Number of blocks */
    size_t numRowsBlocks = _nVectors / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + (_nVectors - numRowsBlocks * numRowsInBlock);

    /* Check if input data are already normalized */
    if (inputTable.isNormalized(NumericTableIface::standardScoreNormalized))
    {
        SafeStatus safeStat;
        /* In case of non-inplace just copy input array to output */
        if (&inputTable != &resultTable)
        {
            daal::threader_for(numRowsBlocks, numRowsBlocks, [&](int iRowsBlock) {
                size_t _nRows    = (iRowsBlock < (numRowsBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
                size_t _startRow = iRowsBlock * numRowsInBlock;

                ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(inputTable, _startRow, _nRows);
                DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
                const algorithmFPType * dataArray_local = dataTableBD.get();

                WriteOnlyRows<algorithmFPType, cpu, NumericTable> normDataTableBD(resultTable, _startRow, _nRows);
                DAAL_CHECK_BLOCK_STATUS_THR(normDataTableBD);
                algorithmFPType * normDataArray_local = normDataTableBD.get();

                for (size_t i = 0; i < _nRows; i++)
                {
                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < _nFeatures; j++)
                    {
                        normDataArray_local[i * _nFeatures + j] = dataArray_local[i * _nFeatures + j];
                    }
                }
            });

            resultTable.setNormalizationFlag(NumericTableIface::standardScoreNormalized);
        }

        return safeStat.detach();
    }

    const daal::algorithms::normalization::zscore::interface3::BaseParameter * const par =
        static_cast<const daal::algorithms::normalization::zscore::interface3::BaseParameter *>(&parameter);
    const bool doScale = par->doScale;

    SafeStatus safeStat;

    /* Call method-specific function to compute means and variances */
    Status s;
    DAAL_CHECK_STATUS(s, computeMeanVariance_thr(inputTable, mean_total, variances_total, parameter));

    if (doScale)
    {
        TArrayCalloc<algorithmFPType, cpu> invSigmas(_nFeatures);
        DAAL_CHECK_MALLOC(invSigmas.get());
        for (size_t j = 0; j < _nFeatures; ++j)
        {
            if (variances_total[j]) invSigmas[j] = algorithmFPType(1.0) / MathInst<algorithmFPType, cpu>::sSqrt(variances_total[j]);
        }
        /* Final normalization threaded loop */
        daal::threader_for(numRowsBlocks, numRowsBlocks, [&](int iRowsBlock) {
            size_t _nRows    = (iRowsBlock < (numRowsBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
            size_t _startRow = iRowsBlock * numRowsInBlock;

            ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(inputTable, _startRow, _nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
            const algorithmFPType * dataArray_local = dataTableBD.get();

            WriteOnlyRows<algorithmFPType, cpu, NumericTable> normDataTableBD(resultTable, _startRow, _nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(normDataTableBD);
            algorithmFPType * normDataArray_local = normDataTableBD.get();

            for (size_t i = 0; i < _nRows; i++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < _nFeatures; j++)
                {
                    normDataArray_local[i * _nFeatures + j] = (algorithmFPType)(dataArray_local[i * _nFeatures + j] - mean_total[j]) * invSigmas[j];
                }
            }
        });
    }
    else
    {
        /* Final normalization threaded loop */
        daal::threader_for(numRowsBlocks, numRowsBlocks, [&](int iRowsBlock) {
            size_t _nRows    = (iRowsBlock < (numRowsBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
            size_t _startRow = iRowsBlock * numRowsInBlock;

            ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(inputTable, _startRow, _nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
            const algorithmFPType * dataArray_local = dataTableBD.get();

            WriteOnlyRows<algorithmFPType, cpu, NumericTable> normDataTableBD(resultTable, _startRow, _nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(normDataTableBD);
            algorithmFPType * normDataArray_local = normDataTableBD.get();

            for (size_t i = 0; i < _nRows; i++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < _nFeatures; j++)
                {
                    normDataArray_local[i * _nFeatures + j] = dataArray_local[i * _nFeatures + j] - mean_total[j];
                }
            }
        });
    }

    resultTable.setNormalizationFlag(NumericTableIface::standardScoreNormalized);

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status ZScoreKernelBase<algorithmFPType, cpu>::compute(NumericTable & inputTable, NumericTable & resultTable,
                                                       const daal::algorithms::Parameter & parameter)
{
    const size_t _nFeatures = inputTable.getNumberOfColumns();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nFeatures, sizeof(algorithmFPType));

    /* Internal arrays for mean and variance, initialized by zeros */
    TArrayCalloc<algorithmFPType, cpu> meanTotal(_nFeatures);
    algorithmFPType * mean_total = meanTotal.get();
    DAAL_CHECK(mean_total, ErrorMemoryAllocationFailed);

    TArrayCalloc<algorithmFPType, cpu> variancesTotal(_nFeatures);
    algorithmFPType * variances_total = variancesTotal.get();
    DAAL_CHECK(variances_total, ErrorMemoryAllocationFailed);

    return common_compute(inputTable, resultTable, mean_total, variances_total, parameter);
}

template <typename algorithmFPType, CpuType cpu>
Status ZScoreKernelBase<algorithmFPType, cpu>::compute(NumericTable & inputTable, NumericTable & resultTable, NumericTable & resultMeans,
                                                       NumericTable & resultVariances, const daal::algorithms::Parameter & parameter)
{
    const size_t _nFeatures = inputTable.getNumberOfColumns();
    const daal::algorithms::normalization::zscore::interface3::BaseParameter * par =
        static_cast<const daal::algorithms::normalization::zscore::interface3::BaseParameter *>(&parameter);

    bool computeMeans     = par->resultsToCompute & mean;
    bool computeVariances = par->resultsToCompute & variances;

    if ((computeMeans) || (computeVariances))
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nFeatures, sizeof(algorithmFPType));
    }

    /* Internal arrays for mean and variance, initialized by zeros */
    TArray<algorithmFPType, cpu> meansTotalArray(computeMeans ? 0 : _nFeatures);
    TArray<algorithmFPType, cpu> variancesTotalArray(computeVariances ? 0 : _nFeatures);

    WriteOnlyRows<algorithmFPType, cpu> meansTotal;
    WriteOnlyRows<algorithmFPType, cpu> variancesTotal;

    algorithmFPType * mean_total = computeMeans ? meansTotal.set(resultMeans, 0, _nFeatures) : meansTotalArray.get();

    DAAL_CHECK(mean_total, ErrorMemoryAllocationFailed);

    algorithmFPType * variances_total = computeVariances ? variancesTotal.set(resultVariances, 0, _nFeatures) : variancesTotalArray.get();

    DAAL_CHECK(variances_total, ErrorMemoryAllocationFailed);

    return common_compute(inputTable, resultTable, mean_total, variances_total, parameter);
};

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
