/* file: low_order_moments_estimates_online.i */
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

/* Common structure with arrays and variables */
template <typename algorithmFPType, CpuType cpu>
struct common_moments_data_t
{
    common_moments_data_t(NumericTable * dataTable) : dataTable(dataTable), prevSums(nullptr), mean(nullptr), variance(nullptr), firstRow(nullptr) {}

    Status init(PartialResult * partialResult, bool isOnline)
    {
        nVectors   = dataTable->getNumberOfRows();
        nFeatures  = dataTable->getNumberOfColumns();
        int result = 0;

        dataTable->getBlockOfRows(0, 1, readOnly, dataBD);
        firstRow = dataBD.getBlockPtr();
        if (!firstRow) return Status(services::ErrorMemoryAllocationFailed);

        ReadWriteMode rwMode = (isOnline ? readWrite : writeOnly);
        for (size_t i = 0; i < lastPartialResultId + 1; i++)
        {
            resultTable[i] = partialResult->get((PartialResultId)i);
            resultTable[i]->getBlockOfRows(0, 1, rwMode, resultBD[i]);
            resultArray[i] = resultBD[i].getBlockPtr();
            if (!resultArray[i]) return Status(services::ErrorMemoryAllocationFailed);
        }

        if (!isOnline)
        {
            resultArray[(int)nObservations][0] = 0.0;
        }

        const size_t rowSize = nFeatures * sizeof(algorithmFPType);

#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
        mean     = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(rowSize);
        variance = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(rowSize);
        if (!mean || !variance) return Status(services::ErrorMemoryAllocationFailed);
#endif

        prevSums = nullptr;
        if (isOnline)
        {
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
            prevSums = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(rowSize);
            if (!prevSums) return Status(services::ErrorMemoryAllocationFailed);

            result = daal::services::internal::daal_memcpy_s(prevSums, rowSize, resultArray[(int)partialSum], rowSize);
#endif
        }
        return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
    }

    ~common_moments_data_t()
    {
        dataTable->releaseBlockOfRows(dataBD);
        for (size_t i = 0; i < lastPartialResultId + 1; i++)
        {
            resultTable[i]->releaseBlockOfRows(resultBD[i]);
        }

#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
        daal_free(mean);
        daal_free(variance);
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
        if (prevSums)
        {
            daal_free(prevSums);
        }
#endif
    }

    size_t nVectors;
    size_t nFeatures;

    NumericTable * dataTable;
    NumericTablePtr resultTable[lastPartialResultId + 1];

    BlockDescriptor<algorithmFPType> dataBD;
    BlockDescriptor<algorithmFPType> resultBD[lastPartialResultId + 1];

    algorithmFPType * firstRow;
    algorithmFPType * resultArray[lastPartialResultId + 1];

    algorithmFPType * mean;
    algorithmFPType * variance;
    algorithmFPType * prevSums;
};

/* TLS structure with local arrays and variables */
template <typename algorithmFPType, CpuType cpu>
struct tls_moments_data_t
{
    int malloc_errors;
    algorithmFPType nvectors;

#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
    algorithmFPType * mean;
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
    algorithmFPType * sum;
#endif
#if (defined _SUM2_ENABLE_) || (defined _SORM_ENABLE_)
    algorithmFPType * sum2;
#endif
#if (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
    algorithmFPType * varc;
#endif
#if (defined _MIN_ENABLE_)
    algorithmFPType * min;
#endif
#if (defined _MAX_ENABLE_)
    algorithmFPType * max;
#endif

    tls_moments_data_t(const size_t nFeatures)
    {
        malloc_errors = 0;
        nvectors      = 0;

#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
        mean = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
        sum = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if (defined _SUM2_ENABLE_) || (defined _SORM_ENABLE_)
        sum2 = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
        varc = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if (defined _MIN_ENABLE_)
        min = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if (defined _MAX_ENABLE_)
        max = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif

        if (
#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
            (!mean) ||
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
            (!sum) ||
#endif
#if (defined _SUM2_ENABLE_) || (defined _SORM_ENABLE_)
            (!sum2) ||
#endif
#if (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
            (!varc) ||
#endif
#if (defined _MIN_ENABLE_)
            (!min) ||
#endif
#if (defined _MAX_ENABLE_)
            (!max) ||
#endif
            false)
        {
            malloc_errors++;
            return;
        }

/* Initialize min and max arrays */
#if defined _MIN_ENABLE_ || defined _MAX_ENABLE_
        const algorithmFPType maxVal = daal::services::internal::MaxVal<algorithmFPType>::get();
    #ifdef _MIN_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(min, maxVal, nFeatures);
    #endif
    #ifdef _MAX_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(max, -maxVal, nFeatures);
    #endif
#endif /* #if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_) */
    }

    ~tls_moments_data_t()
    {
#if (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_)
        if (mean)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(mean);
            mean = 0;
        }
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
        if (sum)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(sum);
            sum = 0;
        }
#endif
#if (defined _SUM2_ENABLE_) || (defined _SORM_ENABLE_)
        if (sum2)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(sum2);
            sum2 = 0;
        }
#endif
#if (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
        if (varc)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(varc);
            varc = 0;
        }
#endif
#if (defined _MIN_ENABLE_)
        if (min)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(min);
            min = 0;
        }
#endif
#if (defined _MAX_ENABLE_)
        if (max)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(max);
            max = 0;
        }
#endif
    }
};

template <typename algorithmFPType, Method method, CpuType cpu>
Status compute_estimates(NumericTable * dataTable, PartialResult * partialResult, bool isOnline)
{
    common_moments_data_t<algorithmFPType, cpu> _cd(dataTable);
    Status s = _cd.init(partialResult, isOnline);
    DAAL_CHECK_STATUS_VAR(s)

    DAAL_ASSERT(_cd.resultArray[(int)nObservations][0] >= 0)
    const size_t nObs = (size_t)(_cd.resultArray[(int)nObservations][0]);

    /* "Short names" for result arrays */
    algorithmFPType * _min      = _cd.resultArray[(int)partialMinimum];
    algorithmFPType * _max      = _cd.resultArray[(int)partialMaximum];
    algorithmFPType * _sums     = _cd.resultArray[(int)partialSum];
    algorithmFPType * _sumSq    = _cd.resultArray[(int)partialSumSquares];
    algorithmFPType * _sumSqCen = _cd.resultArray[(int)partialSumSquaresCentered];

    if (!isOnline)
    {
#if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_)
        const algorithmFPType maxVal = daal::services::internal::MaxVal<algorithmFPType>::get();
    #ifdef _MIN_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(_min, maxVal, _cd.nFeatures);
    #endif
    #ifdef _MAX_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(_max, -maxVal, _cd.nFeatures);
    #endif
#endif /* (defined _MIN_ENABLE_ || defined _MAX_ENABLE_) */

#if (defined _SUM2_ENABLE_) || (defined _SORM_ENABLE_)
        daal::services::internal::service_memset<algorithmFPType, cpu>(_sumSq, algorithmFPType(0), _cd.nFeatures);
#endif
    }

#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_) || (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
    daal::services::internal::service_memset<algorithmFPType, cpu>(_sums, algorithmFPType(0), _cd.nFeatures);
    daal::services::internal::service_memset<algorithmFPType, cpu>(_cd.variance, algorithmFPType(0), _cd.nFeatures);
    daal::services::internal::service_memset<algorithmFPType, cpu>(_cd.mean, algorithmFPType(0), _cd.nFeatures);
#endif

    /* Rows and features splitting by blocks */
    const size_t blockSize          = getDefaultBatchModeBlockSize<cpu>(_cd.nVectors);
    const size_t numRowsInBlock     = (_cd.nVectors > blockSize) ? blockSize : _cd.nVectors;
    const size_t numRowsBlocks      = _cd.nVectors / numRowsInBlock;
    const size_t numRowsInLastBlock = numRowsInBlock + (_cd.nVectors - numRowsBlocks * numRowsInBlock);

    DAAL_PROFILER_TASK(LowOrderMomentsOnlineTask.compute);
    /* TLS buffers initialization */
    daal::tls<tls_moments_data_t<algorithmFPType, cpu> *> tls_data([&]() { return new tls_moments_data_t<algorithmFPType, cpu>(_cd.nFeatures); });

    SafeStatus safeStat;
    {
        DAAL_PROFILER_TASK(LowOrderMomentsOnlineTask.ProcessBlocks);
        /* Compute partial results for each TLS buffer */
        daal::threader_for(numRowsBlocks, numRowsBlocks, [&](int iBlock) {
            struct tls_moments_data_t<algorithmFPType, cpu> * _td = tls_data.local();
            if (_td->malloc_errors)
            {
                return;
            }

            const size_t _startRow = iBlock * numRowsInBlock;
            const size_t _nRows    = (iBlock < (numRowsBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;

            daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(dataTable, _startRow, _nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
            const algorithmFPType * _dataArray_block = dataTableBD.get();

            for (size_t i = 0; i < _nRows; i++)
            {
                /* loop invariants */
#if defined _MEAN_ENABLE_ || defined _SORM_ENABLE_
                const algorithmFPType _invN = algorithmFPType(1.0) / algorithmFPType(_td->nvectors + 1);
#endif
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < _cd.nFeatures; j++)
                {
                    const algorithmFPType arg = _dataArray_block[i * _cd.nFeatures + j];

#if defined _SUM2_ENABLE_ || defined _SORM_ENABLE_
                    const algorithmFPType arg2 = arg * arg;
#endif
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    const algorithmFPType delta = arg - _td->mean[j];
#endif
#ifdef _MIN_ENABLE_
                    _td->min[j] = arg < _td->min[j] ? arg : _td->min[j];
#endif
#ifdef _MAX_ENABLE_
                    _td->max[j] = arg > _td->max[j] ? arg : _td->max[j];
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
                    _td->sum[j] += arg;
#endif
#ifdef _SUM2_ENABLE_
                    _td->sum2[j] += arg2;
#endif
#ifdef _MEAN_ENABLE_
                    _td->mean[j] += delta * _invN;
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    _td->varc[j] += delta * (arg - _td->mean[j]);
#endif
                }

                _td->nvectors++;
            }
        });
    } // end for DAAL_PROFILER_TASK(LowOrderMomentsOnlineTask.ProcessBlocks);

    {
        DAAL_PROFILER_TASK(LowOrderMomentsOnlineTask.MergeBlocks);
        /* Number of already merged values */
        algorithmFPType n_current = 0;

        bool bMemoryAllocationFailed = false;

        /* Merge results by TLS buffers */
        tls_data.reduce([&](tls_moments_data_t<algorithmFPType, cpu> * _td) {
            if (_td->malloc_errors)
            {
                bMemoryAllocationFailed = true;
                delete _td;
                return;
            }
            if (!safeStat)
            {
                delete _td;
                return;
            }

            /* loop invariants */
            algorithmFPType n1_p_n2        = n_current + _td->nvectors;
            algorithmFPType n1_m_n2        = n_current * _td->nvectors;
            algorithmFPType delta_scale    = n1_m_n2 / n1_p_n2;
            algorithmFPType mean_scale     = algorithmFPType(1.0) / (n1_p_n2);
            algorithmFPType variance_scale = algorithmFPType(1.0) / (n1_p_n2 - algorithmFPType(1.0));

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < _cd.nFeatures; j++)
            {
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                algorithmFPType delta = _td->mean[j] - _cd.mean[j];
#endif
#ifdef _MIN_ENABLE_
                if (_td->min[j] < _min[j]) _min[j] = _td->min[j]; /* merging _min */
#endif
#ifdef _MAX_ENABLE_
                if (_td->max[j] > _max[j]) _max[j] = _td->max[j]; /* merging _max */
#endif
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
                _sums[j] += _td->sum[j]; /* merging _sums */
#endif
#ifdef _SUM2_ENABLE_
                _sumSq[j] += _td->sum2[j]; /* merging sum2 */
#endif
#if defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                _cd.variance[j] =
                    (_td->varc[j] + _cd.variance[j] * (n_current - 1) + delta * delta * delta_scale) * variance_scale; /* merging variances */
#endif
#ifdef _MEAN_ENABLE_
                _cd.mean[j] = (_cd.mean[j] * n_current + _td->mean[j] * _td->nvectors) * mean_scale; /* merging means */
#endif
            }
            /* Increase number of already merged values */
            n_current += _td->nvectors;

            delete _td;
        });

        if (bMemoryAllocationFailed) return Status(daal::services::ErrorMemoryAllocationFailed);
        DAAL_CHECK_SAFE_STATUS();

        if (isOnline)
        {
#if (defined _SUM_ENABLE_) || (defined _MEAN_ENABLE_)
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < _cd.nFeatures; i++)
            {
                _sums[i] += _cd.prevSums[i];
            }
#endif
        }

#if (defined _SUM2C_ENABLE_) || (defined _VARC_ENABLE_) || (defined _STDEV_ENABLE_) || (defined _VART_ENABLE_)
        if (_cd.nVectors > 0)
        {
            algorithmFPType nVectorsM1 = (algorithmFPType)(_cd.nVectors - 1);
            if (!isOnline)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < _cd.nFeatures; i++)
                {
                    _sumSqCen[i] = _cd.variance[i] * nVectorsM1;
                }
            }
            else /* isOnline */
            {
                if (nObs == 0)
                {
                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < _cd.nFeatures; i++)
                    {
                        _sumSqCen[i] += _cd.variance[i] * nVectorsM1;
                    }
                }
                else /* if (nObs != 0) */
                {
                    algorithmFPType invPrevNVectors = 1.0 / (algorithmFPType)nObs;
                    algorithmFPType invNVectors     = 1.0 / (algorithmFPType)_cd.nVectors;
                    algorithmFPType coeff           = (algorithmFPType)(nObs * _cd.nVectors) / (algorithmFPType)(nObs + _cd.nVectors);

                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < _cd.nFeatures; i++)
                    {
                        algorithmFPType mean1 = _cd.prevSums[i] * invPrevNVectors;
                        algorithmFPType mean2 = (_sums[i] - _cd.prevSums[i]) * invNVectors;

                        algorithmFPType ssqdm2 = _cd.variance[i] * nVectorsM1;

                        _sumSqCen[i] += (ssqdm2 + coeff * (mean1 * mean1 + mean2 * mean2 - 2 * mean1 * mean2));
                    }
                } /* if (nObs != 0) */
            }     /* isOnline */
        }         /* if (_cd.nVectors > 0) */
#endif
    } // end for DAAL_PROFILER_TASK(LowOrderMomentsOnlineTask.MergeBlocks);

    _cd.resultArray[(int)nObservations][0] += (algorithmFPType)(_cd.nVectors);

    return Status();
}
