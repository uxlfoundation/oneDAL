/* file: low_order_moments_batch_impl.i */
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
//  Implementation of LowOrderMomentsBatchKernel
//--
*/

#ifndef __LOW_ORDER_MOMENTS_BATCH_IMPL_I__
#define __LOW_ORDER_MOMENTS_BATCH_IMPL_I__

#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"
#include "src/algorithms/low_order_moments/low_order_moments_impl.i"
#include "src/algorithms/low_order_moments/low_order_moments_csr_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
/// Single-pass CSR kernel for the requested estimates; `sumCSR` takes the sums from the table.
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status computeSinglePassCSR(NumericTable * dataTable, Result * result, const Parameter * parameter)
{
    switch (parameter->estimatesToCompute)
    {
    case estimatesMinMax: return csr::estimates_minmax::computeBatch<algorithmFPType, method, cpu>(dataTable, result);
    case estimatesMeanVariance: return csr::estimates_meanvariance::computeBatch<algorithmFPType, method, cpu>(dataTable, result);
    default /* estimatesAll */: break;
    }
    return csr::estimates_all::computeBatch<algorithmFPType, method, cpu>(dataTable, result);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsBatchKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataTable, Result * result,
                                                                                   const Parameter * parameter)
{
    if (method == defaultDense)
    {
        switch (parameter->estimatesToCompute)
        {
        case estimatesMinMax: return estimates_batch_minmax::compute_estimates<algorithmFPType, cpu>(dataTable, result);
        case estimatesMeanVariance: return estimates_batch_meanvariance::compute_estimates<algorithmFPType, cpu>(dataTable, result);
        default /* estimatesAll */: break;
        }
        return estimates_batch_all::compute_estimates<algorithmFPType, cpu>(dataTable, result);
    }

    if (method == singlePassCSR)
    {
        return computeSinglePassCSR<algorithmFPType, method, cpu>(dataTable, result, parameter);
    }

    // TODO: fastCSR and sumCSR have no kernel of their own yet and run the single-pass one.
    if (method == fastCSR || method == sumCSR)
    {
        return computeSinglePassCSR<algorithmFPType, method, cpu>(dataTable, result, parameter);
    }

    LowOrderMomentsBatchTask<algorithmFPType, cpu> task(dataTable, result);

    if (method == sumDense || method == sumCSR)
    {
        Status s = retrievePrecomputedStatsIfPossible<algorithmFPType, cpu>(task.nFeatures, task.nVectors, dataTable, task.resultArray[(int)sum],
                                                                            task.resultArray[(int)mean]);

        DAAL_CHECK_STATUS_VAR(s)
    }
    Status s = computeSum_Mean_SecondOrderRawMoment_Variance_Variation<algorithmFPType, method, cpu>(
        task.nFeatures, task.nVectors, task.dataBlock, task.resultArray[(int)sum], task.resultArray[(int)mean],
        task.resultArray[(int)secondOrderRawMoment], task.resultArray[(int)variance], task.resultArray[(int)variation]);

    DAAL_CHECK_STATUS_VAR(s)

    const bool isOnline = false;

    /* Compute standard deviation */
    daal::internal::MathInst<algorithmFPType, cpu>::vSqrt(task.nFeatures, task.resultArray[(int)variance], task.resultArray[(int)standardDeviation]);

    s = computeMinMaxAndSumOfSquared<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock, task.resultArray[(int)minimum],
                                                           task.resultArray[(int)maximum], task.resultArray[(int)sumSquares], isOnline);

    computeSumOfSquaredDiffsFromMean<algorithmFPType, cpu>(task.nFeatures, task.nVectors, 0, task.resultArray[(int)variance],
                                                           task.resultArray[(int)sum], task.resultArray[(int)sum],
                                                           task.resultArray[(int)sumSquaresCentered], isOnline);

    return s;
}

} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
