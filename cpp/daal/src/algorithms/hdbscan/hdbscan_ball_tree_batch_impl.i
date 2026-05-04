/* file: hdbscan_ball_tree_batch_impl.i */
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
 * HDBSCAN implementation using a ball tree with Boruvka's MST algorithm.
 *
 * The approach:
 *   1. Build a ball tree (hypersphere partitioning) over the input data
 *   2. For each point, find its k-th nearest neighbor via tree search -> core distances
 *   3. Build MST under Mutual Reachability Distance using Boruvka's algorithm
 *      with ball-tree-accelerated nearest-different-component queries
 *   4. Sort MST + extract clusters via condensed tree + EOM (shared code)
 *
 * Ball tree vs kd-tree: ball tree uses hypersphere bounds instead of axis-aligned
 * bounding boxes. More robust in high dimensions where kd-tree pruning degrades.
 * The lower bound for a query point q to a ball (center c, radius r) is:
 *   max(0, dist(q, c) - r)
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/hdbscan/hdbscan_distance_utils.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

using daal::internal::CpuType;
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;

// =========================================================================
// Ball tree node: center + radius instead of split plane + bounding box
// =========================================================================
template <typename FPType>
struct BallNode
{
    int left;        // -1 for leaf
    int right;       // -1 for leaf
    int pointBegin;
    int pointEnd;
    int centerIdx;   // index of the pivot point (used as approximate center)
    FPType radius;   // max distance from center to any point in this ball
    int componentId; // -1 = mixed components, >= 0 = uniform
};

// =========================================================================
// Build ball tree recursively
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static int buildBallTree(const algorithmFPType * data, int * pointIndices, int begin, int end, int nCols, BallNode<algorithmFPType> * nodes,
                         int & nextNode, int maxLeafSize, const DistFunc & distFunc)
{
    const int nodeIdx                   = nextNode++;
    BallNode<algorithmFPType> & node    = nodes[nodeIdx];
    node.pointBegin                     = begin;
    node.pointEnd                       = end;
    node.componentId                    = -1;
    node.left                           = -1;
    node.right                          = -1;

    const int count = end - begin;

    // Pick pivot as the first point, find farthest from it, then farthest from that
    int pivot1 = pointIndices[begin];
    algorithmFPType bestDist = algorithmFPType(-1);
    int pivot2 = pivot1;
    for (int i = begin; i < end; i++)
    {
        const algorithmFPType d = distFunc.pointDist(data + pivot1 * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > bestDist)
        {
            bestDist = d;
            pivot2   = pointIndices[i];
        }
    }
    bestDist   = algorithmFPType(-1);
    int pivot3 = pivot2;
    for (int i = begin; i < end; i++)
    {
        const algorithmFPType d = distFunc.pointDist(data + pivot2 * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > bestDist)
        {
            bestDist = d;
            pivot3   = pointIndices[i];
        }
    }

    // Use midpoint between pivot2 and pivot3 conceptually; use pivot2 as center index
    node.centerIdx = pivot2;

    // Compute radius
    algorithmFPType maxR = algorithmFPType(0);
    for (int i = begin; i < end; i++)
    {
        const algorithmFPType d = distFunc.pointDist(data + node.centerIdx * nCols, data + pointIndices[i] * nCols, nCols);
        if (d > maxR) maxR = d;
    }
    node.radius = maxR;

    if (count <= maxLeafSize)
    {
        return nodeIdx;
    }

    // Split: partition points by distance to pivot2 vs pivot3
    const algorithmFPType * p2 = data + pivot2 * nCols;
    const algorithmFPType * p3 = data + pivot3 * nCols;

    int mid = begin;
    int hi  = end - 1;
    while (mid <= hi)
    {
        const algorithmFPType d2 = distFunc.pointDist(data + pointIndices[mid] * nCols, p2, nCols);
        const algorithmFPType d3 = distFunc.pointDist(data + pointIndices[mid] * nCols, p3, nCols);
        if (d2 <= d3)
        {
            mid++;
        }
        else
        {
            std::swap(pointIndices[mid], pointIndices[hi]);
            hi--;
        }
    }

    // Ensure both sides are non-empty
    if (mid == begin) mid = begin + 1;
    if (mid == end) mid = end - 1;

    node.left  = buildBallTree(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, distFunc);
    node.right = buildBallTree(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, distFunc);

    return nodeIdx;
}

// =========================================================================
// k-NN query on ball tree
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static void knnQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                             const algorithmFPType * queryPoint, int nodeIdx, KnnHeap<algorithmFPType> & heap, const DistFunc & distFunc)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];

    // Prune: if the closest point in the ball is farther than current k-th NN
    const algorithmFPType distToCenter = distFunc.pointDist(queryPoint, data + node.centerIdx * nCols, nCols);
    const algorithmFPType lowerBound   = (distToCenter > node.radius) ? (distToCenter - node.radius) : algorithmFPType(0);
    if (lowerBound >= heap.maxDist()) return;

    if (node.left < 0)
    {
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi                = pointIndices[i];
            const algorithmFPType dist  = distFunc.pointDist(queryPoint, data + pi * nCols, nCols);
            heap.push(dist, pi);
        }
        return;
    }

    // Visit nearer child first
    const algorithmFPType dLeft  = distFunc.pointDist(queryPoint, data + nodes[node.left].centerIdx * nCols, nCols);
    const algorithmFPType dRight = distFunc.pointDist(queryPoint, data + nodes[node.right].centerIdx * nCols, nCols);

    if (dLeft <= dRight)
    {
        knnQueryBallTree(data, nCols, nodes, pointIndices, queryPoint, node.left, heap, distFunc);
        knnQueryBallTree(data, nCols, nodes, pointIndices, queryPoint, node.right, heap, distFunc);
    }
    else
    {
        knnQueryBallTree(data, nCols, nodes, pointIndices, queryPoint, node.right, heap, distFunc);
        knnQueryBallTree(data, nCols, nodes, pointIndices, queryPoint, node.left, heap, distFunc);
    }
}

// =========================================================================
// Compute minimum core distance per ball tree node
// =========================================================================
template <typename algorithmFPType>
static algorithmFPType computeMinCoreDistsBallTree(const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                                                   const algorithmFPType * coreDistances, algorithmFPType * minCoreDistNode, int nodeIdx)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.left < 0)
    {
        algorithmFPType minCD = std::numeric_limits<algorithmFPType>::max();
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const algorithmFPType cd = coreDistances[pointIndices[i]];
            if (cd < minCD) minCD = cd;
        }
        minCoreDistNode[nodeIdx] = minCD;
        return minCD;
    }
    const algorithmFPType leftMin  = computeMinCoreDistsBallTree(nodes, pointIndices, coreDistances, minCoreDistNode, node.left);
    const algorithmFPType rightMin = computeMinCoreDistsBallTree(nodes, pointIndices, coreDistances, minCoreDistNode, node.right);
    minCoreDistNode[nodeIdx]       = (leftMin < rightMin) ? leftMin : rightMin;
    return minCoreDistNode[nodeIdx];
}

// =========================================================================
// Update component IDs on ball tree nodes
// =========================================================================
template <typename algorithmFPType>
static int updateNodeComponentsBallTree(BallNode<algorithmFPType> * nodes, const int * pointIndices, const int * componentOf, int nodeIdx)
{
    BallNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.left < 0)
    {
        const int firstComp = componentOf[pointIndices[node.pointBegin]];
        for (int i = node.pointBegin + 1; i < node.pointEnd; i++)
        {
            if (componentOf[pointIndices[i]] != firstComp)
            {
                node.componentId = -1;
                return -1;
            }
        }
        node.componentId = firstComp;
        return firstComp;
    }
    const int leftComp  = updateNodeComponentsBallTree(nodes, pointIndices, componentOf, node.left);
    const int rightComp = updateNodeComponentsBallTree(nodes, pointIndices, componentOf, node.right);
    if (leftComp >= 0 && leftComp == rightComp)
    {
        node.componentId = leftComp;
        return leftComp;
    }
    node.componentId = -1;
    return -1;
}

// =========================================================================
// Boruvka nearest-different-component query on ball tree
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static void nearestMrdBoruvkaQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes,
                                           const int * pointIndices, const algorithmFPType * coreDistances,
                                           const algorithmFPType * minCoreDistNode, const int * componentOf,
                                           const algorithmFPType * queryPoint, int queryIdx, algorithmFPType queryCoreD, int queryComponent,
                                           int nodeIdx, algorithmFPType & bestMrd, int & bestIdx, const DistFunc & distFunc)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.componentId == queryComponent) return;

    // Ball-tree MRD lower bound: max(dist_to_center - radius, 0) then max with cores
    const algorithmFPType distToCenter = distFunc.pointDist(queryPoint, data + node.centerIdx * nCols, nCols);
    algorithmFPType mrdLB              = (distToCenter > node.radius) ? (distToCenter - node.radius) : algorithmFPType(0);
    if (queryCoreD > mrdLB) mrdLB = queryCoreD;
    if (minCoreDistNode[nodeIdx] > mrdLB) mrdLB = minCoreDistNode[nodeIdx];
    if (mrdLB >= bestMrd) return;

    if (node.left < 0)
    {
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi = pointIndices[i];
            if (componentOf[pi] == queryComponent) continue;

            const algorithmFPType dist = distFunc.pointDist(queryPoint, data + pi * nCols, nCols);
            algorithmFPType mrd        = dist;
            if (queryCoreD > mrd) mrd = queryCoreD;
            if (coreDistances[pi] > mrd) mrd = coreDistances[pi];

            if (mrd < bestMrd)
            {
                bestMrd = mrd;
                bestIdx = pi;
            }
        }
        return;
    }

    // Visit nearer child first
    const algorithmFPType dLeft  = distFunc.pointDist(queryPoint, data + nodes[node.left].centerIdx * nCols, nCols);
    const algorithmFPType dRight = distFunc.pointDist(queryPoint, data + nodes[node.right].centerIdx * nCols, nCols);

    const int nearChild = (dLeft <= dRight) ? node.left : node.right;
    const int farChild  = (dLeft <= dRight) ? node.right : node.left;

    nearestMrdBoruvkaQueryBallTree(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint, queryIdx,
                                   queryCoreD, queryComponent, nearChild, bestMrd, bestIdx, distFunc);
    nearestMrdBoruvkaQueryBallTree(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint, queryIdx,
                                   queryCoreD, queryComponent, farChild, bestMrd, bestIdx, distFunc);
}

// =========================================================================
// Core distances + Boruvka MST using ball tree
// =========================================================================
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void computeCoreDistAndMstBallTree(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples,
                                          BallNode<algorithmFPType> * nodes, int * pointIndices, int totalTreeNodes,
                                          algorithmFPType * coreDistances, int * mstFrom, int * mstTo, algorithmFPType * mstWeights,
                                          const DistFunc & distFunc)
{
    const int k = static_cast<int>(minSamples);

    // Step 2: Core distances via k-NN on ball tree
    {
        daal::TlsMem<algorithmFPType, cpu> tlsHeapDists(k);
        daal::TlsMem<int, cpu> tlsHeapIndices(k);

        daal::threader_for(static_cast<int>(nRows), static_cast<int>(nRows), [&](size_t i) {
            algorithmFPType * heapDists = tlsHeapDists.local();
            int * heapIndices           = tlsHeapIndices.local();
            if (!heapDists || !heapIndices) return;

            KnnHeap<algorithmFPType> heap;
            heap.init(heapDists, heapIndices, k);

            knnQueryBallTree(data, static_cast<int>(nCols), nodes, pointIndices, data + i * nCols, 0, heap, distFunc);

            coreDistances[i] = heap.maxDist();
        });
    }

    // Per-node minimum core distances
    daal::services::internal::TArray<algorithmFPType, cpu> minCoreDistNodeArr(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeArr.get();
    if (!minCoreDistNode) return;
    computeMinCoreDistsBallTree(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // Step 3: Boruvka MST
    daal::services::internal::TArray<int, cpu> ufParentArr(nRows);
    daal::services::internal::TArray<int, cpu> ufRankArr(nRows);
    daal::services::internal::TArray<int, cpu> componentOfArr(nRows);
    int * ufParent    = ufParentArr.get();
    int * ufRank      = ufRankArr.get();
    int * componentOf = componentOfArr.get();
    if (!ufParent || !ufRank || !componentOf) return;

    daal::services::internal::TArray<algorithmFPType, cpu> pointBestMrdArr(nRows);
    daal::services::internal::TArray<int, cpu> pointBestIdxArr(nRows);
    algorithmFPType * pointBestMrd = pointBestMrdArr.get();
    int * pointBestIdx             = pointBestIdxArr.get();
    if (!pointBestMrd || !pointBestIdx) return;

    daal::services::internal::TArray<algorithmFPType, cpu> compBestMrdArr(nRows);
    daal::services::internal::TArray<int, cpu> compBestFromArr(nRows);
    daal::services::internal::TArray<int, cpu> compBestToArr(nRows);
    algorithmFPType * compBestMrd = compBestMrdArr.get();
    int * compBestFrom            = compBestFromArr.get();
    int * compBestTo              = compBestToArr.get();
    if (!compBestMrd || !compBestFrom || !compBestTo) return;

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]    = static_cast<int>(i);
        ufRank[i]      = 0;
        componentOf[i] = static_cast<int>(i);
    }

    auto ufFind = [&](int x) -> int {
        while (ufParent[x] != x)
        {
            ufParent[x] = ufParent[ufParent[x]];
            x           = ufParent[x];
        }
        return x;
    };
    auto ufUnion = [&](int rx, int ry) {
        if (ufRank[rx] < ufRank[ry])
            ufParent[rx] = ry;
        else if (ufRank[rx] > ufRank[ry])
            ufParent[ry] = rx;
        else
        {
            ufParent[ry] = rx;
            ufRank[rx]++;
        }
    };

    updateNodeComponentsBallTree(nodes, pointIndices, componentOf, 0);

    size_t edgesAdded    = 0;
    size_t numComponents = nRows;
    const int iNRows     = static_cast<int>(nRows);
    const int iNCols     = static_cast<int>(nCols);

    while (numComponents > 1)
    {
        daal::threader_for(iNRows, iNRows, [&](size_t i) {
            const int comp                   = componentOf[i];
            const algorithmFPType * queryPt  = data + i * nCols;
            const algorithmFPType queryCoreD = coreDistances[i];
            algorithmFPType bestMrd          = std::numeric_limits<algorithmFPType>::max();
            int bestIdx                      = -1;

            nearestMrdBoruvkaQueryBallTree(data, iNCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPt,
                                           static_cast<int>(i), queryCoreD, comp, 0, bestMrd, bestIdx, distFunc);

            pointBestMrd[i] = bestMrd;
            pointBestIdx[i] = bestIdx;
        });

        for (size_t i = 0; i < nRows; i++)
        {
            compBestMrd[i]  = std::numeric_limits<algorithmFPType>::max();
            compBestFrom[i] = -1;
            compBestTo[i]   = -1;
        }
        for (size_t i = 0; i < nRows; i++)
        {
            if (pointBestIdx[i] < 0) continue;
            const int comp = componentOf[i];
            if (pointBestMrd[i] < compBestMrd[comp])
            {
                compBestMrd[comp]  = pointBestMrd[i];
                compBestFrom[comp] = static_cast<int>(i);
                compBestTo[comp]   = pointBestIdx[i];
            }
        }

        size_t addedThisRound = 0;
        for (size_t c = 0; c < nRows; c++)
        {
            if (compBestFrom[c] < 0) continue;
            const int u  = compBestFrom[c];
            const int v  = compBestTo[c];
            const int ru = ufFind(u);
            const int rv = ufFind(v);
            if (ru == rv) continue;

            mstFrom[edgesAdded]    = u;
            mstTo[edgesAdded]      = v;
            mstWeights[edgesAdded] = compBestMrd[c];
            edgesAdded++;
            addedThisRound++;

            ufUnion(ru, rv);
            numComponents--;
        }

        if (addedThisRound == 0) break;

        for (size_t i = 0; i < nRows; i++)
            componentOf[i] = ufFind(static_cast<int>(i));

        updateNodeComponentsBallTree(nodes, pointIndices, componentOf, 0);
    }
}

// =========================================================================
// Main HDBSCAN ball-tree compute
// =========================================================================
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           int metric, double degree, int clusterSelection,
                                                                           bool allowSingleCluster, double clusterSelectionEpsilon,
                                                                           size_t maxClusterSize, double alpha, size_t leafSize)
{
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nCols     = ntData->getNumberOfColumns();
    const size_t edgeCount = nRows - 1;

    if (nRows < 2 || minClusterSize < 2)
    {
        WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(assignBlock);
        int * assignments = assignBlock.get();
        for (size_t i = 0; i < nRows; i++) assignments[i] = -1;
        WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(ncBlock);
        ncBlock.get()[0] = 0;
        return services::Status();
    }

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // Step 1: Build ball tree
    const int maxLeafSize = static_cast<int>(leafSize);
    const int maxNodes    = 4 * static_cast<int>(nRows);

    daal::services::internal::TArray<BallNode<algorithmFPType>, cpu> nodesArr(maxNodes);
    BallNode<algorithmFPType> * nodes = nodesArr.get();
    DAAL_CHECK_MALLOC(nodes);

    daal::services::internal::TArray<int, cpu> pointIndicesArr(nRows);
    int * pointIndices = pointIndicesArr.get();
    DAAL_CHECK_MALLOC(pointIndices);
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<int>(i);

    daal::services::internal::TArray<algorithmFPType, cpu> coreDists(nRows);
    algorithmFPType * coreDistances = coreDists.get();
    DAAL_CHECK_MALLOC(coreDistances);

    daal::services::internal::TArray<int, cpu> mstFromArr(edgeCount);
    daal::services::internal::TArray<int, cpu> mstToArr(edgeCount);
    daal::services::internal::TArray<algorithmFPType, cpu> mstWeightsArr(edgeCount);
    int * mstFrom                = mstFromArr.get();
    int * mstTo                  = mstToArr.get();
    algorithmFPType * mstWeights = mstWeightsArr.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    const bool useAlpha = (alpha != 1.0);

    // Build tree and run core dist + MST, dispatched by metric
    // We need to build the tree with the same distance functor used for queries
    #define DISPATCH_BALL_TREE(DIST_FUNC) \
    { \
        int nextNode = 0; \
        buildBallTree(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodes, nextNode, maxLeafSize, DIST_FUNC); \
        const int totalTreeNodes = nextNode; \
        computeCoreDistAndMstBallTree<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, \
                                                            coreDistances, mstFrom, mstTo, mstWeights, DIST_FUNC); \
    }

    using FP = algorithmFPType;
    using Eucl = EuclideanDist<FP>;
    using Manh = ManhattanDist<FP>;
    using Mink = MinkowskiDist<FP>;
    using Cheb = ChebyshevDist<FP>;
    using AlphaManh = AlphaScaledDist<FP, Manh>;
    using AlphaMink = AlphaScaledDist<FP, Mink>;
    using AlphaCheb = AlphaScaledDist<FP, Cheb>;
    using AlphaEucl = AlphaScaledDist<FP, Eucl>;

    switch (metric)
    {
    case manhattan:
        if (useAlpha)
            DISPATCH_BALL_TREE(AlphaManh(Manh(), alpha))
        else
            DISPATCH_BALL_TREE(Manh())
        break;
    case minkowski:
        if (useAlpha)
            DISPATCH_BALL_TREE(AlphaMink(Mink(degree), alpha))
        else
            DISPATCH_BALL_TREE(Mink(degree))
        break;
    case chebyshev:
        if (useAlpha)
            DISPATCH_BALL_TREE(AlphaCheb(Cheb(), alpha))
        else
            DISPATCH_BALL_TREE(Cheb())
        break;
    default: // euclidean
        if (useAlpha)
            DISPATCH_BALL_TREE(AlphaEucl(Eucl(), alpha))
        else
            DISPATCH_BALL_TREE(Eucl())
        break;
    }

    #undef DISPATCH_BALL_TREE

    // Steps 4-5: Sort MST + Extract clusters (shared with brute_force/kd_tree)
    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                        clusterSelection, allowSingleCluster, clusterSelectionEpsilon,
                                                                        maxClusterSize);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
