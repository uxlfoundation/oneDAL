/* file: low_order_moments_online_impl.i */
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
//  Implementation of LowOrderMomentsOnlineKernel
//--
*/

#ifndef __LOW_ORDER_MOMENTS_ONLINE_IMPL_I__
#define __LOW_ORDER_MOMENTS_ONLINE_IMPL_I__

#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"
#include "src/algorithms/low_order_moments/low_order_moments_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
using namespace daal::services;
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsOnlineKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataTable, PartialResult * partialResult,
                                                                                    const Parameter * parameter, bool isOnline)
{
    if (method == defaultDense)
    {
        switch (parameter->estimatesToCompute)
        {
        case estimatesMinMax: return estimates_online_minmax::compute_estimates<algorithmFPType, method, cpu>(dataTable, partialResult, isOnline);
        case estimatesMeanVariance:
            return estimates_online_meanvariance::compute_estimates<algorithmFPType, method, cpu>(dataTable, partialResult, isOnline);
        default /* estimatesAll */: break;
        }
        return estimates_online_all::compute_estimates<algorithmFPType, method, cpu>(dataTable, partialResult, isOnline);
    }

    LowOrderMomentsOnlineTask<algorithmFPType, cpu> task(dataTable);
    Status s;
    DAAL_CHECK_STATUS(s, task.init(partialResult, isOnline));
    if (method == sumDense || method == sumCSR)
    {
        s = retrievePrecomputedStatsIfPossible<algorithmFPType, cpu>(task.nFeatures, task.nVectors, dataTable, task.resultArray[(int)partialSum],
                                                                     task.mean);

        DAAL_CHECK_STATUS_VAR(s)
    }

    s = computeSumAndVariance<algorithmFPType, method, cpu>(task.nFeatures, task.nVectors, task.dataBlock, task.resultArray[(int)partialSum],
                                                            task.prevSums, task.mean, task.raw2Mom, task.variance, isOnline);

    DAAL_CHECK_STATUS_VAR(s)

    s = computeMinMaxAndSumOfSquared<algorithmFPType, cpu>(task.nFeatures, task.nVectors, task.dataBlock, task.resultArray[(int)partialMinimum],
                                                           task.resultArray[(int)partialMaximum], task.resultArray[(int)partialSumSquares], true);

    computeSumOfSquaredDiffsFromMean<algorithmFPType, cpu>(task.nFeatures, task.nVectors, (size_t)(task.resultArray[(int)nObservations][0]),
                                                           task.variance, task.resultArray[(int)partialSum], task.prevSums,
                                                           task.resultArray[(int)partialSumSquaresCentered], isOnline);

    task.resultArray[(int)nObservations][0] += (algorithmFPType)(task.nVectors);

    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(
    NumericTable * nObservationsTable, NumericTable * sumTable, NumericTable * sumSqTable, NumericTable * sumSqCenTable, NumericTable * meanTable,
    NumericTable * raw2MomTable, NumericTable * varianceTable, NumericTable * stDevTable, NumericTable * variationTable, const Parameter * parameter)
{
    LowOrderMomentsFinalizeTask<algorithmFPType, cpu> task(nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable,
                                                           varianceTable, stDevTable, variationTable);

    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsOnlineTask.finalize);
    finalize<algorithmFPType, cpu>(task);
    return Status();
}

} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
