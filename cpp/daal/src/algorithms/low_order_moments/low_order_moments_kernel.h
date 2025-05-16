/* file: low_order_moments_kernel.h */
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
//  Declaration of template function that calculate low order moments.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_KERNEL_H__
#define __LOW_ORDER_MOMENTS_KERNEL_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/moments/low_order_moments_types.h"
#include "src/services/service_profiler.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
template <typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, Result * result, const Parameter * parameter);
};

template <typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsOnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, PartialResult * partialResult, const Parameter * parameter, bool isOnline);

    services::Status finalizeCompute(NumericTable * nObservationsTable, NumericTable * sumTable, NumericTable * sumSqTable,
                                     NumericTable * sumSqCenTable, NumericTable * meanTable, NumericTable * raw2MomTable,
                                     NumericTable * varianceTable, NumericTable * stDevTable, NumericTable * variationTable,
                                     const Parameter * parameter);
};

template <typename algorithmFPType, low_order_moments::Method method, CpuType cpu>
class LowOrderMomentsDistributedKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(data_management::DataCollection * partialResultsCollection, PartialResult * partialResult, const Parameter * parameter);

    services::Status finalizeCompute(NumericTable * nObservationsTable, NumericTable * sumTable, NumericTable * sumSqTable,
                                     NumericTable * sumSqCenTable, NumericTable * meanTable, NumericTable * raw2MomTable,
                                     NumericTable * varianceTable, NumericTable * stDevTable, NumericTable * variationTable,
                                     const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsBatchTask
{
    LowOrderMomentsBatchTask(NumericTable * dataTable, Result * result);
    virtual ~LowOrderMomentsBatchTask();

    size_t nVectors;
    size_t nFeatures;

    NumericTable * dataTable;
    NumericTablePtr resultTable[lastResultId + 1];

    BlockDescriptor<algorithmFPType> dataBD;
    BlockDescriptor<algorithmFPType> resultBD[lastResultId + 1];

    algorithmFPType * dataBlock;
    algorithmFPType * resultArray[lastResultId + 1];
};

template <typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsOnlineTask
{
    LowOrderMomentsOnlineTask(NumericTable * dataTable);
    virtual ~LowOrderMomentsOnlineTask();
    Status init(PartialResult * partialResult, bool isOnline);

    size_t nVectors;
    size_t nFeatures;

    NumericTable * dataTable;
    NumericTablePtr resultTable[lastPartialResultId + 1];

    BlockDescriptor<algorithmFPType> dataBD;
    BlockDescriptor<algorithmFPType> resultBD[lastPartialResultId + 1];

    algorithmFPType * dataBlock;
    algorithmFPType * resultArray[lastPartialResultId + 1];

    algorithmFPType * mean;
    algorithmFPType * raw2Mom;
    algorithmFPType * variance;
    algorithmFPType * stDev;
    algorithmFPType * variation;
    algorithmFPType * prevSums;
};

template <typename algorithmFPType, CpuType cpu>
struct LowOrderMomentsFinalizeTask
{
    LowOrderMomentsFinalizeTask(NumericTable * nObservationsTable, NumericTable * sumTable, NumericTable * sumSqTable, NumericTable * sumSqCenTable,
                                NumericTable * meanTable, NumericTable * raw2MomTable, NumericTable * varianceTable, NumericTable * stDevTable,
                                NumericTable * variationTable);
    virtual ~LowOrderMomentsFinalizeTask();

    size_t nFeatures;

    NumericTable * nObservationsTable;
    NumericTable * sumTable;
    NumericTable * sumSqTable;
    NumericTable * sumSqCenTable;

    NumericTable * meanTable;
    NumericTable * raw2MomTable;
    NumericTable * varianceTable;
    NumericTable * stDevTable;
    NumericTable * variationTable;

    BlockDescriptor<int> nObservationsBD;
    BlockDescriptor<algorithmFPType> sumBD;
    BlockDescriptor<algorithmFPType> sumSqBD;
    BlockDescriptor<algorithmFPType> sumSqCenBD;

    BlockDescriptor<algorithmFPType> meanBD;
    BlockDescriptor<algorithmFPType> raw2MomBD;
    BlockDescriptor<algorithmFPType> varianceBD;
    BlockDescriptor<algorithmFPType> stDevBD;
    BlockDescriptor<algorithmFPType> variationBD;

    int * nObservations;
    algorithmFPType * sums;
    algorithmFPType * sumSq;
    algorithmFPType * sumSqCen;

    algorithmFPType * mean;
    algorithmFPType * raw2Mom;
    algorithmFPType * variance;
    algorithmFPType * stDev;
    algorithmFPType * variation;
};

} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
