/* file: dbscan_kernel.h */
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
//  Declaration of template function that computes DBSCAN.
//--
*/

#ifndef __DBSCAN_KERNEL_H
#define __DBSCAN_KERNEL_H

#include "algorithms/dbscan/dbscan_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dbscan/dbscan_utils.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANBatchKernel : public Kernel
{
public:
    services::Status computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                      NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                      const Parameter * par);

    services::Status computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                    NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                    const Parameter * par);

private:
    services::Status processNeighborhood(size_t clusterId, int * assignments, const Neighborhood<algorithmFPType, cpu> & neigh,
                                         Queue<size_t, cpu> & qu);

    services::Status processNeighborhoodParallel(size_t clusterId, int * const assignments, const Neighborhood<algorithmFPType, cpu> & neigh,
                                                 daal::tls<Queue<size_t, cpu> *> & tls, TArray<Neighborhood<algorithmFPType, cpu>, cpu> & neighs,
                                                 algorithmFPType minObservations, int * const isCore, size_t nestedLevel);

    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore, const NumericTable * ntData,
                                             NumericTable * ntCoreIndices, NumericTable * ntCoreObservations);
};

template <typename algorithmFPType, CpuType cpu>
class DBSCANBatchKernel<algorithmFPType, kdTree, cpu> : public Kernel
{
public:
    services::Status computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                      NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                      const Parameter * par);

    services::Status computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                    NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                    const Parameter * par);
};

template <typename algorithmFPType, CpuType cpu>
class DBSCANBatchKernel<algorithmFPType, ballTree, cpu> : public Kernel
{
public:
    services::Status computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                      NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                      const Parameter * par);

    services::Status computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                    NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                    const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep1Kernel : public Kernel
{
public:
    services::Status compute(const NumericTable * ntData, NumericTable * ntPartialOrder, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep2Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialData, NumericTable * ntBoundingBox, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep3Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialData, const DataCollection * dcPartialBoundingBoxes, NumericTable * ntSplit,
                             const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep4Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialData, const DataCollection * dcPartialSplits, const DataCollection * dcPartialOrders,
                             DataCollection * dcPartitionedData, DataCollection * dcPartitionedPartialOrders, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep5Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialData, const DataCollection * dcPartialBoundingBoxes,
                             DataCollection * dcPartitionedHaloData, DataCollection * dcPartitionedHaloDataIndices, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep6Kernel : public Kernel
{
public:
    services::Status computeNoMemSave(const DataCollection * dcPartialData, const DataCollection * dcHaloData,
                                      const DataCollection * dcHaloDataIndices, const DataCollection * dcHaloBlocks,
                                      NumericTable * ntClusterStructure, NumericTable * ntFinishedFlag, NumericTable * ntNClusters,
                                      DataCollection * dcQueries, const Parameter * par);

    services::Status computeMemSave(const DataCollection * dcPartialData, const DataCollection * dcHaloData, const DataCollection * dcHaloDataIndices,
                                    const DataCollection * dcHaloBlocks, NumericTable * ntClusterStructure, NumericTable * ntFinishedFlag,
                                    NumericTable * ntNClusters, DataCollection * dcQueries, const Parameter * par);

private:
    services::Status processNeighborhood(size_t clusterId, size_t startObs, int * const clusterStructure,
                                         const Neighborhood<algorithmFPType, cpu> & neigh, Queue<size_t, cpu> & qu);

    services::Status processHaloNeighborhood(size_t startObs, int * const haloAssignments, const int * const haloBlocks,
                                             const int * const haloDataIndices, const Neighborhood<algorithmFPType, cpu> & haloNeigh,
                                             Vector<int, cpu> * const queries);

    services::Status generateQueries(size_t blockIndex, size_t nBlocks, Vector<int, cpu> * const queries, DataCollection * const dcQueries,
                                     bool & totalFinishedFlag);

    template <typename T>
    services::Status repackIntoSingleNT(const DataCollection * dcInput, NumericTablePtr & ntOutput);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep7Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialFinishedFlags, NumericTable * ntFinishedFlag);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep8Kernel : public Kernel
{
public:
    services::Status compute(const NumericTable * ntInputClusterStructure, const NumericTable * ntInputNClusters,
                             const DataCollection * dcPartialQueries, NumericTable * ntClusterStructure, NumericTable * ntFinishedFlag,
                             NumericTable * ntNClusters, DataCollection * dcQueries, const Parameter * par);

private:
    services::Status addQuery(Vector<int, cpu> * const queries, size_t dstblockIndex, size_t dstId, size_t srcblockIndex, size_t srcId);

    services::Status sortQueries(Vector<int, cpu> & queries);

    services::Status removeDuplicateQueries(Vector<int, cpu> & queries, size_t & nUniqueQueries);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep9Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialNClusters, DataCollection * dcClusterOffsets);

    services::Status finalizeCompute(const DataCollection * dcClusterOffsets, NumericTable * ntNClusters);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep10Kernel : public Kernel
{
public:
    services::Status compute(const NumericTable * ntInputClusterStructure, const NumericTable * ntClusterOffset, NumericTable * ntClusterStructure,
                             NumericTable * ntFinishedFlag, DataCollection * dcQueries, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep11Kernel : public Kernel
{
public:
    services::Status compute(const NumericTable * ntInputClusterStructure, const DataCollection * dcPartialQueries, NumericTable * ntClusterStructure,
                             NumericTable * ntFinishedFlag, DataCollection * dcQueries, const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep12Kernel : public Kernel
{
public:
    services::Status compute(NumericTable * ntInputClusterStructure, const DataCollection * dcPartialOrders, DataCollection * dcAssignmentQueries,
                             const Parameter * par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class DBSCANDistrStep13Kernel : public Kernel
{
public:
    services::Status compute(const DataCollection * dcPartialAssignmentQueries, NumericTable * ntAssignmentQueries, const Parameter * par);

    services::Status finalizeCompute(const NumericTable * ntAssignmentQueries, NumericTable * ntAssignments, const Parameter * par);
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
