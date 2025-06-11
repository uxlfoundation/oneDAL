/* file: service_stat.h */
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
//  Template wrappers for STAT functions.
//--
*/

#ifndef __SERVICE_STAT_H__
#define __SERVICE_STAT_H__

#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_blas.h"

#include "src/externals/config.h"

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl>
struct Statistics
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;
    typedef typename _impl<fpType, cpu>::MethodType MethodType;
    typedef typename _impl<fpType, cpu>::ErrorType ErrorType;

    static ErrorType xcp(fpType * data, SizeType nFeatures, SizeType nVectors, fpType * nPreviousObservations, fpType * sum, fpType * crossProduct,
                         MethodType method)
    {
        return _impl<fpType, cpu>::xcp(data, nFeatures, nVectors, nPreviousObservations, sum, crossProduct, method);
    }

    static ErrorType xxcp_weight(fpType * data, SizeType nFeatures, SizeType nVectors, fpType * weight, fpType * accumWeight, fpType * mean,
                                 fpType * crossProduct, MethodType method)
    {
        return _impl<fpType, cpu>::xxcp_weight(data, nFeatures, nVectors, weight, accumWeight, mean, crossProduct, method);
    }

    static ErrorType xxvar_weight(fpType * data, SizeType nFeatures, SizeType nVectors, fpType * weight, fpType * accumWeight, fpType * mean,
                                  fpType * sampleVariance, MethodType method)
    {
        return _impl<fpType, cpu>::xxvar_weight(data, nFeatures, nVectors, weight, accumWeight, mean, sampleVariance, method);
    }

    static ErrorType x2c_mom(const fpType * data, const SizeType nFeatures, const SizeType nVectors, fpType * variance, const MethodType method)
    {
        return _impl<fpType, cpu>::x2c_mom(data, nFeatures, nVectors, variance, method);
    }

    static ErrorType xoutlierdetection(const fpType * data, const SizeType nFeatures, const SizeType nVectors, const SizeType nParams,
                                       const fpType * baconParams, fpType * baconWeights)
    {
        return _impl<fpType, cpu>::xoutlierdetection(data, nFeatures, nVectors, nParams, baconParams, baconWeights);
    }

    static ErrorType xLowOrderMoments(fpType * data, SizeType nFeatures, SizeType nVectors, MethodType method, fpType * sum, fpType * mean,
                                      fpType * secondRawMoment, fpType * variance, fpType * variation)
    {
        return _impl<fpType, cpu>::xLowOrderMoments(data, nFeatures, nVectors, method, sum, mean, secondRawMoment, variance, variation);
    }

    static ErrorType xSumAndVariance(fpType * data, SizeType nFeatures, SizeType nVectors, fpType * nPreviousObservations, MethodType method,
                                     fpType * sum, fpType * mean, fpType * secondRawMoment, fpType * variance)
    {
        return _impl<fpType, cpu>::xSumAndVariance(data, nFeatures, nVectors, nPreviousObservations, method, sum, mean, secondRawMoment, variance);
    }

    static ErrorType xQuantiles(const fpType * data, const SizeType nFeatures, const SizeType nVectors, const SizeType quantOrderN,
                                const fpType * quantOrder, fpType * quants)
    {
        return _impl<fpType, cpu>::xQuantiles(data, nFeatures, nVectors, quantOrderN, quantOrder, quants);
    }

    static ErrorType xSort(fpType * data, SizeType nFeatures, SizeType nVectors, fpType * sortedData)
    {
        return _impl<fpType, cpu>::xSort(data, nFeatures, nVectors, sortedData);
    }

    // works with column-major layout. Now works faster than MKL version for any layout, so it is used in EM until MKL optimizations are completed
    static void xxcp_weight_byrows(const fpType * weights, const fpType * data, size_t nRows, size_t nCols, fpType * dataWeightedBuffer, fpType & sum,
                                   fpType * weightedSum, fpType * weightedCP)
    {
        // calculate W_X: W_X(i,j) = w(i) * X(i,j)
        // calculate weighted sums
        for (size_t j = 0; j < nCols; j++)
        {
            const fpType * dataCol = data + j * nRows;
            fpType * dataWeightCol = dataWeightedBuffer + j * nRows;

            fpType wsum = 0;

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            PRAGMA_ICC_NO16(omp simd reduction(+ : wsum))
            for (size_t i = 0; i < nRows; i++)
            {
                dataWeightCol[i] = weights[i] * dataCol[i];
                wsum += dataWeightCol[i];
            }
            weightedSum[j] = wsum;
        }

        // calculate W_all = sum of weights
        fpType sum_local = 0;
        PRAGMA_ICC_NO16(omp simd reduction(+ : sum_local))
        for (size_t i = 0; i < nRows; i++)
        {
            sum_local += weights[i];
        }

        sum = sum_local;

        const fpType invSum = 1 / (sum);

        // calcultate weightd means from sums
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nCols; i++)
        {
            weightedSum[i] = weightedSum[i] * invSum;
        }

        // calculate cross-product as cp = W_X(T) * X
        char transa  = 'T';
        char transb  = 'N';
        fpType alpha = 1.0;
        fpType beta  = 0.0;
        DAAL_INT n   = nRows;
        DAAL_INT p   = nCols;
        BlasInst<fpType, cpu>::xxgemm(&transa, &transb, &p, &p, &n, &alpha, dataWeightedBuffer, &n, data, &n, &beta, weightedCP, &p);

        // calculate covariance as COV = cp - W_all * mean(T) * mean
        alpha = -sum;
        beta  = 1.0;
        n     = 1;
        BlasInst<fpType, cpu>::xxgemm(&transa, &transb, &p, &p, &n, &alpha, weightedSum, &n, weightedSum, &n, &beta, weightedCP, &p);
    }
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using StatisticsInst = Statistics<fpType, cpu, StatisticsBackend>;
} // namespace internal
} // namespace daal

#endif
