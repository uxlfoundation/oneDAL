/* file: dbscan_ball_tree_batch_impl.i */
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
 * DBSCAN implementation using a ball tree for accelerated epsilon-neighborhood queries.
 *
 * Ball tree uses hypersphere-bounded spatial partitioning, which is more robust
 * in high dimensions than axis-aligned kd-tree splits.
 *
 * The approach:
 *   1. Build a ball tree over the input data
 *   2. For each point, find all neighbors within epsilon via tree range search
 *   3. Mark core points (those with >= min_observations neighbors)
 *   4. Expand clusters from core points using BFS with tree-accelerated queries
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
struct DbscanBallNode
{
    int left;
    int right;
    int pointBegin;
    int pointEnd;
    int centerIdx;
    FPType radius;
};

template <typename FPType>
static FPType euclideanDist(const FPType * a, const FPType * b, int nCols)
{
    FPType sum = FPType(0);
    for (int d = 0; d < nCols; d++)
    {
        const FPType diff = a[d] - b[d];
        sum += diff * diff;
    }
    return static_cast<FPType>(sqrt(static_cast<double>(sum)));
}

template <typename FPType>
static int buildDbscanBallTree(const FPType * data, int * pointIndices, int begin, int end, int nCols, DbscanBallNode<FPType> * nodes, int & nextNode,
                               int maxLeafSize)
{
    const int nodeIdx             = nextNode++;
    DbscanBallNode<FPType> & node = nodes[nodeIdx];
    node.pointBegin               = begin;
    node.pointEnd                 = end;
    node.left                     = -1;
    node.right                    = -1;

    const int count = end - begin;

    // Find diameter approximation: pick first, find farthest, find farthest from that
    int pivot1       = pointIndices[begin];
    FPType bestDist  = FPType(-1);
    int pivot2       = pivot1;
    for (int i = begin; i < end; i++)
    {
        const FPType d = euclideanDist(data + pivot1 * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > bestDist)
        {
            bestDist = d;
            pivot2   = pointIndices[i];
        }
    }
    bestDist   = FPType(-1);
    int pivot3 = pivot2;
    for (int i = begin; i < end; i++)
    {
        const FPType d = euclideanDist(data + pivot2 * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > bestDist)
        {
            bestDist = d;
            pivot3   = pointIndices[i];
        }
    }

    node.centerIdx = pivot2;

    // Compute radius
    FPType maxR = FPType(0);
    for (int i = begin; i < end; i++)
    {
        const FPType d = euclideanDist(data + node.centerIdx * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > maxR) maxR = d;
    }
    node.radius = maxR;

    if (count <= maxLeafSize)
    {
        return nodeIdx;
    }

    // Split by distance to pivot2 vs pivot3
    const FPType * p2 = data + pivot2 * nCols;
    const FPType * p3 = data + pivot3 * nCols;

    int mid = begin;
    int hi  = end - 1;
    while (mid <= hi)
    {
        const FPType d2 = euclideanDist(data + pointIndices[mid] * nCols, p2, nCols);
        const FPType d3 = euclideanDist(data + pointIndices[mid] * nCols, p3, nCols);
        if (d2 <= d3)
            mid++;
        else
        {
            std::swap(pointIndices[mid], pointIndices[hi]);
            hi--;
        }
    }

    if (mid == begin) mid = begin + 1;
    if (mid == end) mid = end - 1;

    node.left  = buildDbscanBallTree(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize);
    node.right = buildDbscanBallTree(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize);

    return nodeIdx;
}

// Epsilon-range query on ball tree
template <typename FPType>
static void ballTreeRangeQuery(const FPType * data, int nCols, const DbscanBallNode<FPType> * nodes, const int * pointIndices,
                               const FPType * queryPoint, int nodeIdx, FPType epsilon, std::vector<int> & result)
{
    const DbscanBallNode<FPType> & node = nodes[nodeIdx];

    // Prune: if closest point in ball is beyond epsilon
    const FPType distToCenter = euclideanDist(queryPoint, data + node.centerIdx * nCols, nCols);
    const FPType lowerBound   = distToCenter - node.radius;
    if (lowerBound > epsilon) return;

    if (node.left < 0)
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

    // Visit nearer child first
    const FPType dLeft  = euclideanDist(queryPoint, data + nodes[node.left].centerIdx * nCols, nCols);
    const FPType dRight = euclideanDist(queryPoint, data + nodes[node.right].centerIdx * nCols, nCols);

    if (dLeft <= dRight)
    {
        ballTreeRangeQuery(data, nCols, nodes, pointIndices, queryPoint, node.left, epsilon, result);
        ballTreeRangeQuery(data, nCols, nodes, pointIndices, queryPoint, node.right, epsilon, result);
    }
    else
    {
        ballTreeRangeQuery(data, nCols, nodes, pointIndices, queryPoint, node.right, epsilon, result);
        ballTreeRangeQuery(data, nCols, nodes, pointIndices, queryPoint, node.left, epsilon, result);
    }
}

// Count neighbors within epsilon on ball tree
template <typename FPType>
static int ballTreeCountNeighbors(const FPType * data, int nCols, const DbscanBallNode<FPType> * nodes, const int * pointIndices,
                                  const FPType * queryPoint, int nodeIdx, FPType epsilon)
{
    const DbscanBallNode<FPType> & node = nodes[nodeIdx];

    const FPType distToCenter = euclideanDist(queryPoint, data + node.centerIdx * nCols, nCols);
    const FPType lowerBound   = distToCenter - node.radius;
    if (lowerBound > epsilon) return 0;

    // If entire ball is within epsilon, count all points
    if (distToCenter + node.radius <= epsilon)
    {
        return node.pointEnd - node.pointBegin;
    }

    if (node.left < 0)
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

    return ballTreeCountNeighbors(data, nCols, nodes, pointIndices, queryPoint, node.left, epsilon) +
           ballTreeCountNeighbors(data, nCols, nodes, pointIndices, queryPoint, node.right, epsilon);
}

template <typename algorithmFPType, CpuType cpu>
services::Status DBSCANBatchKernel<algorithmFPType, ballTree, cpu>::computeNoMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                                      NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                                      NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                                      const Parameter * par)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();
    const algorithmFPType epsilon = static_cast<algorithmFPType>(par->epsilon);
    const size_t minObservations  = par->minObservations;
    const int leafSize            = 40;

    ReadRows<algorithmFPType, cpu> dataRows(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataRows);
    const algorithmFPType * data = dataRows.get();

    // Allocate tree structures
    const int maxNodes = 2 * static_cast<int>(nRows) + 1;
    TArray<DbscanBallNode<algorithmFPType>, cpu> nodesArr(maxNodes);
    TArray<int, cpu> pointIndicesArr(nRows);
    DAAL_CHECK_MALLOC(nodesArr.get() && pointIndicesArr.get());

    int * pointIndices = pointIndicesArr.get();
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<int>(i);

    int nextNode = 0;
    buildDbscanBallTree(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodesArr.get(), nextNode, leafSize);

    // Phase 1: Identify core points
    TArray<int, cpu> isCoreArr(nRows);
    DAAL_CHECK_MALLOC(isCoreArr.get());
    int * isCore = isCoreArr.get();

    for (size_t i = 0; i < nRows; i++)
    {
        const algorithmFPType * queryPoint = data + i * nCols;
        const int cnt = ballTreeCountNeighbors(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, queryPoint, 0, epsilon);
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
        ballTreeRangeQuery(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, data + i * nCols, 0, epsilon, neighbors);

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
            ballTreeRangeQuery(data, static_cast<int>(nCols), nodesArr.get(), pointIndices, data + pt * nCols, 0, epsilon, neighbors);

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
services::Status DBSCANBatchKernel<algorithmFPType, ballTree, cpu>::computeMemSave(const NumericTable * ntData, const NumericTable * ntWeights,
                                                                                    NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                                    NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                                    const Parameter * par)
{
    return computeNoMemSave(ntData, ntWeights, ntAssignments, ntNClusters, ntCoreIndices, ntCoreObservations, par);
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
