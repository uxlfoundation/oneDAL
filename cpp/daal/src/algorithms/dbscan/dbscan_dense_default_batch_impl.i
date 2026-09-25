/* file: dbscan_dense_default_batch_impl.i */
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
//  Implementation of default method for DBSCAN algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"

#include "algorithms/dbscan/dbscan_types.h"
#include "src/algorithms/dbscan/dbscan_utils.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
#define __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT 64
#define __DBSCAN_MAXIMUM_NESTED_STACK_LEVEL     200

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANBatchKernel<algorithmFPType, method, cpu>::processNeighborhood(size_t clusterId, int * const assignments,
                                                                            const Neighborhood<algorithmFPType, cpu> & neigh, Queue<size_t, cpu> & qu)
{
    for (size_t j = 0; j < neigh.size(); j++)
    {
        const size_t nextObs = neigh.get(j);
        if (assignments[nextObs] == noise)
        {
            assignments[nextObs] = clusterId;
        }
        else if (assignments[nextObs] == undefined)
        {
            assignments[nextObs] = clusterId;
            DAAL_CHECK_STATUS_VAR(qu.push(nextObs));
        }
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANBatchKernel<algorithmFPType, method, cpu>::processNeighborhoodParallel(
    size_t clusterId, int * const assignments, const Neighborhood<algorithmFPType, cpu> & neigh, daal::tls<Queue<size_t, cpu> *> & tls,
    TArray<Neighborhood<algorithmFPType, cpu>, cpu> & neighs, algorithmFPType minObservations, int * const isCore, size_t nestedLevel)
{
    const size_t nBlocks   = threader_get_max_threads_number();
    const size_t blockSize = neigh.size() / nBlocks + !!(neigh.size() % nBlocks);

    SafeStatus safeStat;

    daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
        size_t total_elems = 0;

        size_t begin = iBlock * blockSize;
        size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, neigh.size());

        total_elems += end - begin;

        auto & qu = *tls.local();

        for (size_t j = begin; j < end; j++)
        {
            const size_t nextObs = neigh.get(j);
            if (assignments[nextObs] == noise)
            {
                assignments[nextObs] = clusterId;
            }
            else if (assignments[nextObs] == undefined)
            {
                assignments[nextObs] = clusterId;
                DAAL_CHECK_STATUS_THR(qu.push(nextObs));
            }
        }

        while (!qu.empty())
        {
            const size_t curObs = qu.pop();

            Neighborhood<algorithmFPType, cpu> & curNeigh = neighs[curObs];

            assignments[curObs] = clusterId;
            if (curNeigh.weight() < minObservations) continue;

            isCore[curObs] = 1;

            total_elems += curNeigh.size();

            if (nestedLevel < __DBSCAN_MAXIMUM_NESTED_STACK_LEVEL)
            {
                DAAL_CHECK_STATUS_THR(
                    processNeighborhoodParallel(clusterId, assignments, curNeigh, tls, neighs, minObservations, isCore, nestedLevel + 1));
            }
            else
            {
                DAAL_CHECK_STATUS_THR(processNeighborhood(clusterId, assignments, curNeigh, qu));
            }
        }
    });

    return safeStat.detach();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANBatchKernel<algorithmFPType, method, cpu>::processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore,
                                                                                const NumericTable * ntData, NumericTable * ntCoreIndices,
                                                                                NumericTable * ntCoreObservations)
{
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();

    size_t nCoreObservations = 0;

    for (size_t i = 0; i < nRows; i++)
    {
        if (!isCore[i])
        {
            continue;
        }
        nCoreObservations++;
    }

    if (nCoreObservations == 0)
    {
        return Status();
    }

    if (resultsToCompute & computeCoreIndices)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->resize(nCoreObservations));
        WriteRows<int, cpu> coreIndicesRows(ntCoreIndices, 0, nCoreObservations);
        DAAL_CHECK_BLOCK_STATUS(coreIndicesRows);
        int * const coreIndices = coreIndicesRows.get();

        size_t pos = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            coreIndices[pos] = i;
            pos++;
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->resize(nCoreObservations));
        WriteRows<algorithmFPType, cpu> coreObservationsRows(ntCoreObservations, 0, nCoreObservations);
        DAAL_CHECK_BLOCK_STATUS(coreObservationsRows);
        algorithmFPType * const coreObservations = coreObservationsRows.get();

        size_t pos = 0;
        int result = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            ReadRows<algorithmFPType, cpu> dataRows(const_cast<NumericTable *>(ntData), i, 1);
            DAAL_CHECK_BLOCK_STATUS(dataRows);
            const algorithmFPType * const data = dataRows.get();

            result |= daal::services::internal::daal_memcpy_s(&(coreObservations[pos * nFeatures]), sizeof(algorithmFPType) * nFeatures, data,
                                                              sizeof(algorithmFPType) * nFeatures);
            pos++;
        }
        if (result)
        {
            return Status(services::ErrorMemoryCopyFailedInternal);
        }
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANBatchKernel<algorithmFPType, method, cpu>::computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                         NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                         NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                         const Parameter * par)
{
    Status s;
    const size_t nRows = ntData->getNumberOfRows();

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(Neighborhood<algorithmFPType, cpu>));

    TArray<Neighborhood<algorithmFPType, cpu>, cpu> neighs(nRows);
    DAAL_CHECK_MALLOC(neighs.get());

    NeighborhoodEngine<method, algorithmFPType, cpu> nEngine(ntData, ntData, ntWeights, epsilon, minkowskiPower);
    DAAL_CHECK_STATUS_VAR(nEngine.queryFull(neighs.get()));

    WriteRows<int, cpu> assignRows(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignRows);
    int * const assignments = assignRows.get();

    service_memset<int, cpu>(assignments, undefined, nRows);

    TArray<int, cpu> isCoreArray(nRows);
    DAAL_CHECK_MALLOC(isCoreArray.get());
    int * const isCore = isCoreArray.get();

    service_memset<int, cpu>(isCore, 0, nRows);

    size_t nClusters = 0;

    daal::tls<Queue<size_t, cpu> *> tls([=]() { return new Queue<size_t, cpu>; });

    for (size_t i = 0; i < nRows; i++)
    {
        if (assignments[i] != undefined) continue;

        Neighborhood<algorithmFPType, cpu> & curNeigh = neighs[i];
        if (curNeigh.weight() < minObservations)
        {
            assignments[i] = noise;
            continue;
        }

        nClusters++;
        isCore[i]      = 1;
        assignments[i] = nClusters - 1;

        DAAL_CHECK_STATUS_VAR(processNeighborhoodParallel(nClusters - 1, assignments, curNeigh, tls, neighs, minObservations, isCore, 0));
    }

    tls.reduce([=](Queue<size_t, cpu> * q) { delete q; });

    WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nClustersRows);
    nClustersRows.get()[0] = nClusters;

    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, isCore, ntData, ntCoreIndices, ntCoreObservations));
    }

    const size_t nBlocks   = neighs.size();
    const size_t blockSize = neighs.size() / nBlocks + !!(neighs.size() % nBlocks);
    daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
        size_t begin = iBlock * blockSize;
        size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, neighs.size());

        for (size_t i = begin; i < end; ++i)
        {
            neighs[i].clear();
        }
    });

    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DBSCANBatchKernel<algorithmFPType, method, cpu>::computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                       NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                       NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                       const Parameter * par)
{
    Status s;

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    const size_t nRows = ntData->getNumberOfRows();

    NeighborhoodEngine<method, algorithmFPType, cpu> nEngine(ntData, ntData, ntWeights, epsilon, minkowskiPower);

    WriteRows<int, cpu> assignRows(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignRows);
    int * const assignments = assignRows.get();

    service_memset<int, cpu>(assignments, undefined, nRows);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(int));

    TArray<int, cpu> isCoreArray(nRows);
    DAAL_CHECK_MALLOC(isCoreArray.get());
    int * const isCore = isCoreArray.get();

    service_memset<int, cpu>(isCore, 0, nRows);

    const size_t prefetchBlockSize = __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT;
    TArray<Neighborhood<algorithmFPType, cpu>, cpu> prefetchedNeighs(prefetchBlockSize);
    DAAL_CHECK_MALLOC(prefetchedNeighs.get());

    size_t nClusters = 0;
    Queue<size_t, cpu> qu;

    for (size_t i = 0; i < nRows; i++)
    {
        if (assignments[i] != undefined) continue;

        Neighborhood<algorithmFPType, cpu> curNeigh;
        nEngine.query(&i, 1, &curNeigh);

        if (curNeigh.weight() < minObservations)
        {
            assignments[i] = noise;
            continue;
        }

        nClusters++;
        isCore[i]      = 1;
        assignments[i] = nClusters - 1;

        qu.reset();

        DAAL_CHECK_STATUS_VAR(processNeighborhood(nClusters - 1, assignments, curNeigh, qu));

        size_t firstPrefetchedPos = 0;
        size_t lastPrefetchedPos  = 0;

        while (!qu.empty())
        {
            const size_t quPos  = qu.head();
            const size_t curObs = qu.pop();

            if (quPos >= lastPrefetchedPos)
            {
                firstPrefetchedPos = lastPrefetchedPos = quPos;
            }

            if (lastPrefetchedPos - firstPrefetchedPos < prefetchBlockSize && lastPrefetchedPos < qu.tail())
            {
                const size_t nextPrefetchPos =
                    firstPrefetchedPos + prefetchBlockSize < qu.tail() ? firstPrefetchedPos + prefetchBlockSize : qu.tail();
                DAAL_CHECK_STATUS_VAR(nEngine.query(qu.getInternalPtr(lastPrefetchedPos), nextPrefetchPos - lastPrefetchedPos,
                                                    &(prefetchedNeighs[lastPrefetchedPos - firstPrefetchedPos]), true));
                lastPrefetchedPos = nextPrefetchPos;
            }

            Neighborhood<algorithmFPType, cpu> & curNeigh = prefetchedNeighs[quPos - firstPrefetchedPos];

            assignments[curObs] = nClusters - 1;
            if (curNeigh.weight() < minObservations) continue;

            isCore[curObs] = 1;

            DAAL_CHECK_STATUS_VAR(processNeighborhood(nClusters - 1, assignments, curNeigh, qu));
        }
    }

    WriteRows<int, cpu> nClustersRows(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nClustersRows);
    nClustersRows.get()[0] = nClusters;

    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, isCore, ntData, ntCoreIndices, ntCoreObservations));
    }

    return s;
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
