/* file: low_order_moments_estimates_batch.i */
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

using namespace daal::services;
/* Common structure with arrays and variables */
template <typename algorithmFPType, CpuType cpu>
struct common_moments_data_t
{
    common_moments_data_t(NumericTable * dataTable, Result * r) : dataTable(dataTable)
    {
        malloc_errors = 0;

        nVectors  = dataTable->getNumberOfRows();
        nFeatures = dataTable->getNumberOfColumns();

        dataTable->getBlockOfRows(0, 1 /*nVectors*/, readOnly, firstRowBD);
        firstRow = firstRowBD.getBlockPtr();
        if (!firstRow)
        {
            malloc_errors++;
            return;
        }

        for (size_t i = 0; i < lastResultId + 1; i++)
        {
            resultTable[i] = r->get((ResultId)i);
            resultTable[i]->getBlockOfRows(0, 1, writeOnly, resultBD[i]);
            resultArray[i] = resultBD[i].getBlockPtr();
            if (!(resultArray[i]))
            {
                malloc_errors++;
                return;
            }
        }

#ifdef _MEAN_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)mean], 0, nFeatures);
#endif
#ifdef _SORM_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)secondOrderRawMoment], 0, nFeatures);
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)variance], 0, nFeatures);
#endif
#ifdef _SUM_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)sum], 0, nFeatures);
#endif
#ifdef _SUM2_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)sumSquares], 0, nFeatures);
#endif

#if defined _MIN_ENABLE_ || defined _MAX_ENABLE_
        const algorithmFPType maxVal = daal::services::internal::MaxVal<algorithmFPType>::get();
    #ifdef _MIN_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)minimum], maxVal, nFeatures);
    #endif
    #ifdef _MAX_ENABLE_
        daal::services::internal::service_memset<algorithmFPType, cpu>(resultArray[(int)maximum], -maxVal, nFeatures);
    #endif
#endif /* #if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_) */

        return;
    }

    ~common_moments_data_t()
    {
        dataTable->releaseBlockOfRows(firstRowBD);
        for (size_t i = 0; i < lastResultId + 1; i++)
        {
            resultTable[i]->releaseBlockOfRows(resultBD[i]);
        }
    }

    int malloc_errors;

    size_t nVectors;
    size_t nFeatures;

    NumericTable * dataTable;
    NumericTablePtr resultTable[lastResultId + 1];

    BlockDescriptor<algorithmFPType> firstRowBD;
    BlockDescriptor<algorithmFPType> resultBD[lastResultId + 1];

    algorithmFPType * firstRow;
    algorithmFPType * resultArray[lastResultId + 1];
};

/* TLS structure with local arrays and variables */
template <typename algorithmFPType, CpuType cpu>
struct tls_moments_data_t
{
    int malloc_errors;
    algorithmFPType nvectors;

#ifdef _MEAN_ENABLE_
    algorithmFPType * mean;
#endif
#ifdef _SUM_ENABLE_
    algorithmFPType * sum;
#endif
#ifdef _SUM2_ENABLE_
    algorithmFPType * sum2;
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
    algorithmFPType * varc;
#endif
#ifdef _MIN_ENABLE_
    algorithmFPType * min;
#endif
#ifdef _MAX_ENABLE_
    algorithmFPType * max;
#endif

    tls_moments_data_t(size_t nFeatures)
    {
        malloc_errors = 0;
        nvectors      = 0;

#ifdef _MEAN_ENABLE_
        mean = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _SUM_ENABLE_
        sum = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _SUM2_ENABLE_
        sum2 = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
        varc = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _MIN_ENABLE_
        min = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _MAX_ENABLE_
        max = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif

        if (
#ifdef _MEAN_ENABLE_
            (!mean) ||
#endif
#ifdef _SUM_ENABLE_
            (!sum) ||
#endif
#ifdef _SUM2_ENABLE_
            (!sum2) ||
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
            (!varc) ||
#endif
#ifdef _MIN_ENABLE_
            (!min) ||
#endif
#ifdef _MAX_ENABLE_
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
#ifdef _MEAN_ENABLE_
        if (mean)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(mean);
            mean = 0;
        }
#endif
#ifdef _SUM_ENABLE_
        if (sum)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(sum);
            sum = 0;
        }
#endif
#ifdef _SUM2_ENABLE_
        if (sum2)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(sum2);
            sum2 = 0;
        }
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
        if (varc)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(varc);
            varc = 0;
        }
#endif
#ifdef _MIN_ENABLE_
        if (min)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(min);
            min = 0;
        }
#endif
#ifdef _MAX_ENABLE_
        if (max)
        {
            daal::services::internal::service_scalable_free<algorithmFPType, cpu>(max);
            max = 0;
        }
#endif
    }
};

template <typename algorithmFPType, CpuType cpu>
Status compute_estimates(NumericTable * dataTable, Result * result)
{
    /* Common data structure */
    common_moments_data_t<algorithmFPType, cpu> _cd(dataTable, result);
    if (_cd.malloc_errors) return Status(daal::services::ErrorMemoryAllocationFailed);

    /* Rows and features splitting by blocks */
    const size_t blockSize          = getDefaultBatchModeBlockSize<cpu>(_cd.nVectors);
    const size_t numRowsInBlock     = (_cd.nVectors > blockSize) ? blockSize : _cd.nVectors;
    const size_t numRowsBlocks      = _cd.nVectors / numRowsInBlock;
    const size_t numRowsInLastBlock = numRowsInBlock + (_cd.nVectors - numRowsBlocks * numRowsInBlock);

    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTask.compute);
    /* TLS buffers initialization */
    daal::tls<tls_moments_data_t<algorithmFPType, cpu> *> tls_data([&]() { return new tls_moments_data_t<algorithmFPType, cpu>(_cd.nFeatures); });

    SafeStatus safeStat;
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTask.ProcessBlocks);
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
#if defined _MEAN_ENABLE_
                const algorithmFPType _invN = algorithmFPType(1.0) / algorithmFPType(_td->nvectors + 1);
#endif

                const algorithmFPType * const argi = _dataArray_block + i * _cd.nFeatures;

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < _cd.nFeatures; j++)
                {
                    const algorithmFPType arg = argi[j];
#if (defined _SUM2_ENABLE_ || defined _SORM_ENABLE_)
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

#ifdef _SUM_ENABLE_
                    _td->sum[j] += arg;
#endif
#if (defined _SUM2_ENABLE_ || defined _SORM_ENABLE_)
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
    } /* end for  DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTask.ProcessBlocks); */

    /* Number of already merged values */
    algorithmFPType n_current = 0;

    bool bMemoryAllocationFailed = false;

    /* "Short names" for result arrays */
    algorithmFPType * _min   = _cd.resultArray[(int)minimum];
    algorithmFPType * _max   = _cd.resultArray[(int)maximum];
    algorithmFPType * _sum   = _cd.resultArray[(int)sum];
    algorithmFPType * _sum2  = _cd.resultArray[(int)sumSquares];
    algorithmFPType * _sum2c = _cd.resultArray[(int)sumSquaresCentered];
    algorithmFPType * _mean  = _cd.resultArray[(int)mean];
    algorithmFPType * _sorm  = _cd.resultArray[(int)secondOrderRawMoment];
    algorithmFPType * _varc  = _cd.resultArray[(int)variance];
    algorithmFPType * _stdev = _cd.resultArray[(int)standardDeviation];
    algorithmFPType * _vart  = _cd.resultArray[(int)variation];

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTask.MergeBlocks);
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

#undef _THREAD_REDUCTION_MIN_SIZE_
#define _THREAD_REDUCTION_MIN_SIZE_ 128

            /* Make reduction stage threaded for wide matrices */
            if (_cd.nFeatures >= _THREAD_REDUCTION_MIN_SIZE_)
            {

#undef _FEATURE_BLOCK_SIZE_
#define _FEATURE_BLOCK_SIZE_ 32
                const size_t numFeaturesInBlock     = (_cd.nFeatures > _FEATURE_BLOCK_SIZE_) ? _FEATURE_BLOCK_SIZE_ : _cd.nFeatures;
                const size_t numFeatureBlocks       = _cd.nFeatures / numFeaturesInBlock;
                const size_t numFeaturesInLastBlock = numFeaturesInBlock + (_cd.nFeatures - numFeatureBlocks * numFeaturesInBlock);

                daal::threader_for(numFeatureBlocks, numFeatureBlocks, [&](int iFeatureBlock) {
                    size_t _jstart = iFeatureBlock * numFeaturesInBlock;
                    size_t _jend   = _jstart + ((iFeatureBlock < (numFeatureBlocks - 1)) ? numFeaturesInBlock : numFeaturesInLastBlock);

                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = _jstart; j < _jend; j++)
                    {
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                        algorithmFPType delta = _td->mean[j] - _mean[j];
    #ifdef _MEAN_ENABLE_
                        _mean[j] = (_mean[j] * n_current + _td->mean[j] * _td->nvectors) * mean_scale; /* merging means */
    #endif
#endif

#ifdef _SUM_ENABLE_
                        _sum[j] += _td->sum[j]; /* merging sums */
#endif
#ifdef _SUM2_ENABLE_
                        _sum2[j] += _td->sum2[j]; /* merging sum2 */
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                        _sum2c[j] = (_td->varc[j] + _varc[j] * (n_current - 1) + delta * delta * delta_scale); /* merging _sum2c */
#endif
#if defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                        _varc[j] = _sum2c[j] * variance_scale; /* merging variances */
#endif

#ifdef _MIN_ENABLE_
                        if (_td->min[j] < _min[j]) _min[j] = _td->min[j]; /* merging min */
#endif
#ifdef _MAX_ENABLE_
                        if (_td->max[j] > _max[j]) _max[j] = _td->max[j]; /* merging max */
#endif
                    }
                });
            }    /* if(_cd.nFeatures >= _THREAD_REDUCTION_MIN_SIZE_) */
            else /* if(_cd.nFeatures < _THREAD_REDUCTION_MIN_SIZE_) */
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < _cd.nFeatures; j++)
                {
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    algorithmFPType delta = _td->mean[j] - _mean[j];
#endif
#ifdef _MEAN_ENABLE_
                    _mean[j] = (_mean[j] * n_current + _td->mean[j] * _td->nvectors) * mean_scale; /* merging means */
#endif

#ifdef _SUM_ENABLE_
                    _sum[j] += _td->sum[j]; /* merging sums */
#endif
#ifdef _SUM2_ENABLE_
                    _sum2[j] += _td->sum2[j]; /* merging sum2 */
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    _sum2c[j] = (_td->varc[j] + _varc[j] * (n_current - 1) + delta * delta * delta_scale); /* merging _sum2c */
#endif
#if defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    _varc[j] = _sum2c[j] * variance_scale; /* merging variances */
#endif

#ifdef _MIN_ENABLE_
                    if (_td->min[j] < _min[j]) _min[j] = _td->min[j]; /* merging min */
#endif
#ifdef _MAX_ENABLE_
                    if (_td->max[j] > _max[j]) _max[j] = _td->max[j]; /* merging max */
#endif
                }
            } /* if(_cd.nFeatures < _THREAD_REDUCTION_MIN_SIZE_) */

            /* Increase number of already merged values */
            n_current += _td->nvectors;

            delete _td;
        });

        if (bMemoryAllocationFailed) return Status(daal::services::ErrorMemoryAllocationFailed);

        DAAL_CHECK_SAFE_STATUS();

        /* Final loop for std deviations and variations */
#if (defined _STDEV_ENABLE_ || defined _VART_ENABLE_)
        daal::internal::MathInst<algorithmFPType, cpu>::vSqrt(_cd.nFeatures, &_varc[0], &_stdev[0]);
#endif /* #if (defined _STDEV_ENABLE_ || defined _VART_ENABLE_) */

#if (defined _VART_ENABLE_ || defined _SORM_ENABLE_)
        const algorithmFPType _invN = algorithmFPType(1.0) / algorithmFPType(_cd.nVectors);
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < _cd.nFeatures; j++)
        {
    #ifdef _SORM_ENABLE_
            _sorm[j] = _sum2[j] * _invN;
    #endif
    #ifdef _VART_ENABLE_
            _vart[j] = _stdev[j] / _mean[j];
    #endif
        }
#endif /* #if (defined _VART_ENABLE_ || defined _SORM_ENABLE_) */
    }  /* end for DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTask.MergeBlocks); */

    return Status();
} /* compute_estimates */
