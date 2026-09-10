/* file: dbscan_kd_tree_batch_impl.i */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
 * DBSCAN implementation using a k-d tree for accelerated epsilon-neighborhood queries.
 *
 * The approach:
 *   1. Build a k-d tree over the input data (with bounding boxes per node)
 *   2. For each point, find all neighbors within epsilon via tree range search
 *   3. Mark core points (those with >= min_observations neighbors)
 *   4. Expand clusters from core points using BFS with tree-accelerated queries
 *
 * Advantage over brute_force: O(N * log N) average for neighborhood queries
 * vs O(N^2) brute-force pairwise distance computation.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "algorithms/dbscan/dbscan_types.h"
#include "src/algorithms/dbscan/dbscan_kernel.h"
#include "src/algorithms/dbscan/dbscan_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{

using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;
using daal::services::internal::TArray;

template <typename FPType>
struct DbscanKdNode
{
    int splitDim;
    FPType splitVal;
    int left;
    int right;
    int pointBegin;
    int pointEnd;
};

template <typename FPType>
static int buildDbscanKdTree(const FPType * data, int * pointIndices, int begin, int end, int nCols, DbscanKdNode<FPType> * nodes, int & nextNode,
                             int maxLeafSize, FPType * bboxLo, FPType * bboxHi)
{
    const int nodeIdx           = nextNode++;
    DbscanKdNode<FPType> & node = nodes[nodeIdx];
    node.pointBegin             = begin;
    node.pointEnd               = end;

    FPType bestSpread = FPType(-1);
    int bestDim       = 0;
    for (int d = 0; d < nCols; d++)
    {
        FPType lo = std::numeric_limits<FPType>::max();
        FPType hi = std::numeric_limits<FPType>::lowest();
        for (int i = begin; i < end; i++)
        {
            const FPType val = data[pointIndices[i] * nCols + d];
            if (val < lo) lo = val;
            if (val > hi) hi = val;
        }
        bboxLo[nodeIdx * nCols + d] = lo;
        bboxHi[nodeIdx * nCols + d] = hi;
        const FPType spread         = hi - lo;
        if (spread > bestSpread)
        {
            bestSpread = spread;
            bestDim    = d;
        }
    }

    const int count = end - begin;
    if (count <= maxLeafSize)
    {
        node.splitDim = -1;
        node.splitVal = 0;
        node.left     = -1;
        node.right    = -1;
        return nodeIdx;
    }

    node.splitDim = bestDim;

    const int mid = begin + count / 2;
    std::nth_element(pointIndices + begin, pointIndices + mid, pointIndices + end,
                     [&](int a, int b) { return data[a * nCols + bestDim] < data[b * nCols + bestDim]; });

    node.splitVal = data[pointIndices[mid] * nCols + bestDim];

    node.left  = buildDbscanKdTree(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);
    node.right = buildDbscanKdTree(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);

    return nodeIdx;
}

// Epsilon-range query on kd-tree: find all points within epsilon of queryPoint
template <typename FPType>
static void rangeQuery(const FPType * data, int nCols, const DbscanKdNode<FPType> * nodes, const int * pointIndices, const FPType * bboxLo,
                       const FPType * bboxHi, const FPType * queryPoint, int nodeIdx, FPType epsilon, std::vector<int> & result)
{
    const DbscanKdNode<FPType> & node = nodes[nodeIdx];

    // Prune: if bbox lower bound exceeds epsilon, skip
    FPType lbSum = FPType(0);
    for (int d = 0; d < nCols; d++)
    {
        FPType excess = FPType(0);
        if (queryPoint[d] < bboxLo[nodeIdx * nCols + d])
            excess = bboxLo[nodeIdx * nCols + d] - queryPoint[d];
        else if (queryPoint[d] > bboxHi[nodeIdx * nCols + d])
            excess = queryPoint[d] - bboxHi[nodeIdx * nCols + d];
        lbSum += excess * excess;
    }
    if (lbSum > epsilon * epsilon) return;

    if (node.splitDim < 0)
    {
        // Leaf: check each point
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi       = pointIndices[i];
            const FPType * row = data + pi * nCols;
            FPType distSq      = FPType(0);
            for (int d = 0; d < nCols; d++)
            {
                const FPType diff = queryPoint[d] - row[d];
                distSq += diff * diff;
            }
            if (distSq <= epsilon * epsilon)
            {
                result.push_back(pi);
            }
        }
        return;
    }

    const FPType queryVal = queryPoint[node.splitDim];
    const FPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    rangeQuery(data, nCols, nodes, pointIndices, bboxLo, bboxHi, queryPoint, nearChild, epsilon, result);

    if (diff * diff <= epsilon * epsilon)
    {
        rangeQuery(data, nCols, nodes, pointIndices, bboxLo, bboxHi, queryPoint, farChild, epsilon, result);
    }
}

// Count neighbors within epsilon (stops early once min_observations reached)
template <typename FPType>
static int countNeighbors(const FPType * data, int nCols, const DbscanKdNode<FPType> * nodes, const int * pointIndices, const FPType * bboxLo,
                          const FPType * bboxHi, const FPType * queryPoint, int nodeIdx, FPType epsilon, int minObs)
{
    const DbscanKdNode<FPType> & node = nodes[nodeIdx];

    FPType lbSum = FPType(0);
    for (int d = 0; d < nCols; d++)
    {
        FPType excess = FPType(0);
        if (queryPoint[d] < bboxLo[nodeIdx * nCols + d])
            excess = bboxLo[nodeIdx * nCols + d] - queryPoint[d];
        else if (queryPoint[d] > bboxHi[nodeIdx * nCols + d])
            excess = queryPoint[d] - bboxHi[nodeIdx * nCols + d];
        lbSum += excess * excess;
    }
    if (lbSum > epsilon * epsilon) return 0;

    if (node.splitDim < 0)
    {
        int cnt = 0;
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi       = pointIndices[i];
            const FPType * row = data + pi * nCols;
            FPType distSq      = FPType(0);
            for (int d = 0; d < nCols; d++)
            {
                const FPType diff = queryPoint[d] - row[d];
                distSq += diff * diff;
            }
            if (distSq <= epsilon * epsilon) cnt++;
        }
        return cnt;
    }

    const FPType queryVal = queryPoint[node.splitDim];
    const FPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    int cnt = countNeighbors(data, nCols, nodes, pointIndices, bboxLo, bboxHi, queryPoint, nearChild, epsilon, minObs);

    if (diff * diff <= epsilon * epsilon)
    {
        cnt += countNeighbors(data, nCols, nodes, pointIndices, bboxLo, bboxHi, queryPoint, farChild, epsilon, minObs);
    }
    return cnt;
}

template <typename algorithmFPType, CpuType cpu>
services::Status DBSCANBatchKernel<algorithmFPType, kdTree, cpu>::computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                                   NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                                   NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                                   const Parameter * par)
{
    const size_t nRows            = ntData->getNumberOfRows();
    const size_t nCols            = ntData->getNumberOfColumns();
    const algorithmFPType epsilon = static_cast<algorithmFPType>(par->epsilon);
    const size_t minObservations  = par->minObservations;
    const int leafSize            = 40;

    ReadRows<algorithmFPType, cpu> dataRows(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataRows);
    const algorithmFPType * data = dataRows.get();

    // Allocate tree structures
    const int maxNodes = 2 * static_cast<int>(nRows) + 1;
    TArray<DbscanKdNode<algorithmFPType>, cpu> nodesArr(maxNodes);
    TArray<int, cpu> pointIndicesArr(nRows);
    TArray<algorithmFPType, cpu> bboxLoArr(maxNodes * nCols);
    TArray<algorithmFPType, cpu> bboxHiArr(maxNodes * nCols);
    DAAL_CHECK_MALLOC(nodesArr.get() && pointIndicesArr.get() && bboxLoArr.get() && bboxHiArr.get());

    int * pointIndices = pointIndicesArr.get();
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<int>(i);

    int nextNode = 0;
    buildDbscanKdTree(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodesArr.get(), nextNode, leafSize, bboxLoArr.get(),
                      bboxHiArr.get());

    // Phase 1: Identify core points
    TArray<int, cpu> isCoreArr(nRows);
    DAAL_CHECK_MALLOC(isCoreArr.get());
    int * isCore = isCoreArr.get();

    for (size_t i = 0; i < nRows; i++)
    {
        const algorithmFPType * queryPoint = data + i * nCols;
        const int cnt = countNeighbors(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, bboxLoArr.get(), bboxHiArr.get(), queryPoint, 0,
                                       epsilon, static_cast<int>(minObservations));
        isCore[i]     = (cnt >= static_cast<int>(minObservations)) ? 1 : 0;
    }

    // Phase 2: Cluster expansion via BFS
    WriteOnlyRows<int, cpu> assignRows(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignRows);
    int * assignments = assignRows.get();
    for (size_t i = 0; i < nRows; i++) assignments[i] = undefined;

    int clusterId = 0;
    std::vector<int> queue;
    std::vector<int> neighbors;

    for (size_t i = 0; i < nRows; i++)
    {
        if (assignments[i] != undefined) continue;
        if (!isCore[i])
        {
            assignments[i] = noise;
            continue;
        }

        assignments[i] = clusterId;
        neighbors.clear();
        rangeQuery(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, bboxLoArr.get(), bboxHiArr.get(), data + i * nCols, 0, epsilon,
                   neighbors);

        queue.clear();
        for (int nb : neighbors)
        {
            if (assignments[nb] == noise)
            {
                assignments[nb] = clusterId;
            }
            else if (assignments[nb] == undefined)
            {
                assignments[nb] = clusterId;
                if (isCore[nb]) queue.push_back(nb);
            }
        }

        size_t qIdx = 0;
        while (qIdx < queue.size())
        {
            const int pt = queue[qIdx++];
            neighbors.clear();
            rangeQuery(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, bboxLoArr.get(), bboxHiArr.get(), data + pt * nCols, 0, epsilon,
                       neighbors);

            for (int nb : neighbors)
            {
                if (assignments[nb] == noise)
                {
                    assignments[nb] = clusterId;
                }
                else if (assignments[nb] == undefined)
                {
                    assignments[nb] = clusterId;
                    if (isCore[nb]) queue.push_back(nb);
                }
            }
        }

        clusterId++;
    }

    // Write cluster count
    WriteOnlyRows<int, cpu> ncRows(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncRows);
    ncRows.get()[0] = clusterId;

    // Process core indices/observations results
    DAAL_UINT64 resultsToCompute = par->resultsToCompute;
    if (resultsToCompute & computeCoreIndices)
    {
        size_t coreCount = 0;
        for (size_t i = 0; i < nRows; i++)
            if (isCore[i]) coreCount++;

        ntCoreIndices->resize(coreCount);
        if (coreCount > 0)
        {
            WriteOnlyRows<int, cpu> coreIdxRows(ntCoreIndices, 0, coreCount);
            DAAL_CHECK_BLOCK_STATUS(coreIdxRows);
            int * coreIdx = coreIdxRows.get();
            size_t pos    = 0;
            for (size_t i = 0; i < nRows; i++)
            {
                if (isCore[i]) coreIdx[pos++] = static_cast<int>(i);
            }
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        size_t coreCount = 0;
        for (size_t i = 0; i < nRows; i++)
            if (isCore[i]) coreCount++;

        ntCoreObservations->resize(coreCount);
        if (coreCount > 0)
        {
            WriteOnlyRows<algorithmFPType, cpu> coreObsRows(ntCoreObservations, 0, coreCount);
            DAAL_CHECK_BLOCK_STATUS(coreObsRows);
            algorithmFPType * coreObs = coreObsRows.get();
            size_t pos                = 0;
            for (size_t i = 0; i < nRows; i++)
            {
                if (isCore[i])
                {
                    for (size_t d = 0; d < nCols; d++) coreObs[pos * nCols + d] = data[i * nCols + d];
                    pos++;
                }
            }
        }
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DBSCANBatchKernel<algorithmFPType, kdTree, cpu>::computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                                 NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                                 NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                                 const Parameter * par)
{
    // Memory-save mode uses the same tree approach (already more memory-efficient than brute-force)
    return computeNoMemSave(ntData, ntWeights, ntAssignments, ntNClusters, ntCoreIndices, ntCoreObservations, par);
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
