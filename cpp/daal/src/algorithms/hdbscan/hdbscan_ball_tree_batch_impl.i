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
 * HDBSCAN implementation using a ball tree with Boruvka's Minimum Spanning Tree
 * (MST) algorithm: https://arxiv.org/html/2412.07789v1
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

#include "services/daal_atomic_int.h"
#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/hdbscan/hdbscan_distance_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
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
using daal::services::internal::TArray;
using daal::services::internal::TArrayScalable;

// =========================================================================
// Ball tree node: center + radius instead of split plane + bounding box
// =========================================================================
template <typename FPType>
struct BallNode
{
    int left;  // -1 for leaf
    int right; // -1 for leaf
    int pointBegin;
    int pointEnd;
    int centerIdx;   // index of the pivot point (used as approximate center)
    FPType radius;   // max distance from center to any point in this ball
    int componentId; // -1 = mixed components, >= 0 = uniform
};

// =========================================================================
// Gather pointIndices[begin..end) rows into a contiguous scratch buffer so
// all later distance sweeps at this node touch contiguous memory. Paying the
// gather cost once lets pivot2 / pivot3 / radius / partition share one layout
// and avoids repeated FP conversions on low-precision input (e.g. bf16 → FP32).
// =========================================================================
template <typename algorithmFPType>
static void gatherRows(const algorithmFPType * data, const int * pointIndices, int begin, int end, int nCols, algorithmFPType * scratchRows)
{
    const int count = end - begin;
    for (int i = 0; i < count; i++)
    {
        const algorithmFPType * src = data + pointIndices[begin + i] * nCols;
        algorithmFPType * dst       = scratchRows + i * nCols;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (int d = 0; d < nCols; d++) dst[d] = src[d];
    }
}

// =========================================================================
// Compute distances from pivotPt to each of the count rows in scratchRows,
// returning the argmax (position in [0, count)) and filling outDists.
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static int blockDistsAndArgmax(const algorithmFPType * pivotPt, const algorithmFPType * scratchRows, int count, int nCols, const DistFunc & distFunc,
                               algorithmFPType * outDists)
{
    algorithmFPType bestDist = algorithmFPType(-1);
    int argmax               = 0;
    for (int i = 0; i < count; i++)
    {
        const algorithmFPType d = distFunc.pointDist(pivotPt, scratchRows + i * nCols, nCols);
        outDists[i]             = d;
        if (d > bestDist)
        {
            bestDist = d;
            argmax   = i;
        }
    }
    return argmax;
}

// Threshold below which buildBallTree runs its two recursive calls serially.
// Above it, left and right subtrees are spawned as parallel tasks.
// Keeps task-spawn overhead from dominating on small ranges.
constexpr int BALL_TREE_PARALLEL_MIN_RANGE = 2048;

// =========================================================================
// Build ball tree recursively.
//
// Uses three distance sweeps per node (pivot2, pivot3, radius is free) by
// reusing the pivot2-distance array as the node radius pass and keeping the
// d2/d3 partition arrays in lockstep with pointIndices during the swap loop.
//
// `nextNode` is atomic so sibling subtrees, built concurrently, can each claim
// a unique node index. The resulting index order is parent-first but siblings
// may interleave; all traversals go through node.left / node.right, so order
// within the node array is irrelevant apart from the root being index 0.
// =========================================================================
template <typename algorithmFPType, daal::internal::CpuType cpu, typename DistFunc>
static int buildBallTree(const algorithmFPType * data, int * pointIndices, int begin, int end, int nCols, BallNode<algorithmFPType> * nodes,
                         daal::services::AtomicInt & nextNode, int maxLeafSize, const DistFunc & distFunc)
{
    const int nodeIdx                = nextNode.inc() - 1;
    BallNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                  = begin;
    node.pointEnd                    = end;
    node.componentId                 = -1;
    node.left                        = -1;
    node.right                       = -1;

    const int count = end - begin;

    // Gather rows once; all subsequent distance sweeps read contiguous memory.
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> scratchRowsArr(static_cast<size_t>(count) * nCols);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d2Arr(count);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d3Arr(count);
    algorithmFPType * scratchRows = scratchRowsArr.get();
    algorithmFPType * d2          = d2Arr.get();
    algorithmFPType * d3          = d3Arr.get();
    if (!scratchRows || !d2 || !d3) return nodeIdx;

    gatherRows(data, pointIndices, begin, end, nCols, scratchRows);

    // Pick pivot1 as the first point, find farthest from it (pivot2), then farthest
    // from pivot2 (pivot3). Positions are offsets into scratchRows / pointIndices[begin..end).
    // d3 serves as scratch for the pivot1 sweep — it gets recomputed for pivot3 below.
    const int pivot1 = pointIndices[begin];
    const int pos2   = blockDistsAndArgmax(data + pivot1 * nCols, scratchRows, count, nCols, distFunc, d3);
    const int pivot2 = pointIndices[begin + pos2];

    // pivot2 is the ball center; d2[i] = dist(pivot2, scratchRows[i]) feeds both
    // the radius (max) and the partition (compared to d3).
    const int pos3   = blockDistsAndArgmax(data + pivot2 * nCols, scratchRows, count, nCols, distFunc, d2);
    const int pivot3 = pointIndices[begin + pos3];

    node.centerIdx = pivot2;

    algorithmFPType maxR = algorithmFPType(0);
    for (int i = 0; i < count; i++)
    {
        if (d2[i] > maxR) maxR = d2[i];
    }
    node.radius = maxR;

    if (count <= maxLeafSize)
    {
        return nodeIdx;
    }

    // Populate d3 (distances to pivot3), then partition.
    blockDistsAndArgmax(data + pivot3 * nCols, scratchRows, count, nCols, distFunc, d3);

    // Partition points by d2 vs d3. d2/d3/pointIndices swap in lockstep so the
    // distance arrays stay aligned with pointIndices[begin..end) during the sweep.
    int lo = 0;
    int hi = count - 1;
    while (lo <= hi)
    {
        if (d2[lo] <= d3[lo])
        {
            lo++;
        }
        else
        {
            std::swap(pointIndices[begin + lo], pointIndices[begin + hi]);
            std::swap(d2[lo], d2[hi]);
            std::swap(d3[lo], d3[hi]);
            hi--;
        }
    }
    int mid = begin + lo;

    // Ensure both sides are non-empty
    if (mid == begin) mid = begin + 1;
    if (mid == end) mid = end - 1;

    if (count >= BALL_TREE_PARALLEL_MIN_RANGE)
    {
        int leftIdx  = -1;
        int rightIdx = -1;
        daal::task_group tg;
        auto leftFn = [&]() {
            leftIdx = buildBallTree<algorithmFPType, cpu>(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, distFunc);
        };
        auto rightFn = [&]() {
            rightIdx = buildBallTree<algorithmFPType, cpu>(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, distFunc);
        };
        tg.run(leftFn);
        tg.run(rightFn);
        tg.wait();
        node.left  = leftIdx;
        node.right = rightIdx;
    }
    else
    {
        node.left  = buildBallTree<algorithmFPType, cpu>(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, distFunc);
        node.right = buildBallTree<algorithmFPType, cpu>(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, distFunc);
    }

    return nodeIdx;
}

// =========================================================================
// k-NN query on ball tree
// =========================================================================
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void knnQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                             const algorithmFPType * queryPoint, int nodeIdx, KnnHeap<algorithmFPType, cpu> & heap, const DistFunc & distFunc)
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
            const int pi               = pointIndices[i];
            const algorithmFPType dist = distFunc.pointDist(queryPoint, data + pi * nCols, nCols);
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
        algorithmFPType minCD = daal::services::internal::MaxVal<algorithmFPType>::get();
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
static void nearestMrdBoruvkaQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                                           const algorithmFPType * coreDistances, const algorithmFPType * minCoreDistNode, const int * componentOf,
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

    nearestMrdBoruvkaQueryBallTree(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint, queryIdx, queryCoreD,
                                   queryComponent, nearChild, bestMrd, bestIdx, distFunc);
    nearestMrdBoruvkaQueryBallTree(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint, queryIdx, queryCoreD,
                                   queryComponent, farChild, bestMrd, bestIdx, distFunc);
}

// =========================================================================
// Core distances + Boruvka MST using ball tree
// =========================================================================
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void computeCoreDistAndMstBallTree(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples,
                                          BallNode<algorithmFPType> * nodes, int * pointIndices, int totalTreeNodes, algorithmFPType * coreDistances,
                                          int * mstFrom, int * mstTo, algorithmFPType * mstWeights, const DistFunc & distFunc)
{
    const int k = static_cast<int>(minSamples);

    // Step 2: Core distances via k-NN on ball tree
    daal::threader_for(static_cast<int>(nRows), static_cast<int>(nRows), [&](size_t i) {
        KnnHeap<algorithmFPType, cpu> heap(k);
        if (!heap.ok()) return;

        knnQueryBallTree<algorithmFPType, cpu>(data, static_cast<int>(nCols), nodes, pointIndices, data + i * nCols, 0, heap, distFunc);

        coreDistances[i] = heap.maxDist();
    });

    // Per-node minimum core distances
    TArrayScalable<algorithmFPType, cpu> minCoreDistNodeVec(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeVec.get();
    if (!minCoreDistNode) return;
    computeMinCoreDistsBallTree(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // Step 3: Boruvka MST
    TArray<int, cpu> ufParentVec(nRows);
    TArray<int, cpu> ufRankVec(nRows);
    TArray<int, cpu> componentOfVec(nRows);
    int * ufParent    = ufParentVec.get();
    int * ufRank      = ufRankVec.get();
    int * componentOf = componentOfVec.get();
    if (!ufParent || !ufRank || !componentOf) return;

    TArrayScalable<algorithmFPType, cpu> pointBestMrdVec(nRows);
    TArray<int, cpu> pointBestIdxVec(nRows);
    algorithmFPType * pointBestMrd = pointBestMrdVec.get();
    int * pointBestIdx             = pointBestIdxVec.get();
    if (!pointBestMrd || !pointBestIdx) return;

    TArrayScalable<algorithmFPType, cpu> compBestMrdVec(nRows);
    TArray<int, cpu> compBestFromVec(nRows);
    TArray<int, cpu> compBestToVec(nRows);
    algorithmFPType * compBestMrd = compBestMrdVec.get();
    int * compBestFrom            = compBestFromVec.get();
    int * compBestTo              = compBestToVec.get();
    if (!compBestMrd || !compBestFrom || !compBestTo) return;

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]    = static_cast<int>(i);
        componentOf[i] = static_cast<int>(i);
    }
    services::internal::service_memset_seq<int, cpu>(ufRank, 0, nRows);

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
            algorithmFPType bestMrd          = daal::services::internal::MaxVal<algorithmFPType>::get();
            int bestIdx                      = -1;

            nearestMrdBoruvkaQueryBallTree(data, iNCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPt,
                                           static_cast<int>(i), queryCoreD, comp, 0, bestMrd, bestIdx, distFunc);

            pointBestMrd[i] = bestMrd;
            pointBestIdx[i] = bestIdx;
        });

        for (size_t i = 0; i < nRows; i++)
        {
            compBestMrd[i]  = daal::services::internal::MaxVal<algorithmFPType>::get();
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

        daal::threader_for(iNRows, iNRows, [&](size_t i) { componentOf[i] = ufFind(static_cast<int>(i)); });

        updateNodeComponentsBallTree(nodes, pointIndices, componentOf, 0);
    }
}

// =========================================================================
// Helper: build the ball tree and run core-distance + Boruvka MST for a given
// distance functor. Replaces the previous DISPATCH_BALL_TREE macro.
// =========================================================================
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void runBallTreeCoreDistAndMst(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples, int maxLeafSize,
                                      BallNode<algorithmFPType> * nodes, int * pointIndices, algorithmFPType * coreDistances, int * mstFrom,
                                      int * mstTo, algorithmFPType * mstWeights, const DistFunc & distFunc)
{
    daal::services::AtomicInt nextNode(0);
    buildBallTree<algorithmFPType, cpu>(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodes, nextNode, maxLeafSize,
                                        distFunc);
    const int totalTreeNodes = nextNode.get();
    computeCoreDistAndMstBallTree<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, coreDistances, mstFrom,
                                                        mstTo, mstWeights, distFunc);
}

// =========================================================================
// Main HDBSCAN ball-tree compute
// =========================================================================
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           algorithms::internal::PairwiseDistanceType pairwiseDistance,
                                                                           double minkowskiDegree, int clusterSelection, bool allowSingleCluster,
                                                                           double clusterSelectionEpsilon, size_t maxClusterSize, double alpha,
                                                                           size_t leafSize)
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

    TArray<BallNode<algorithmFPType>, cpu> nodesVec(maxNodes);
    BallNode<algorithmFPType> * nodes = nodesVec.get();
    DAAL_CHECK_MALLOC(nodes);

    TArray<int, cpu> pointIndicesVec(nRows);
    int * pointIndices = pointIndicesVec.get();
    DAAL_CHECK_MALLOC(pointIndices);
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<int>(i);

    TArray<algorithmFPType, cpu> coreDistsVec(nRows);
    algorithmFPType * coreDistances = coreDistsVec.get();
    DAAL_CHECK_MALLOC(coreDistances);

    TArray<int, cpu> mstFromVec(edgeCount);
    TArray<int, cpu> mstToVec(edgeCount);
    TArray<algorithmFPType, cpu> mstWeightsVec(edgeCount);
    int * mstFrom                = mstFromVec.get();
    int * mstTo                  = mstToVec.get();
    algorithmFPType * mstWeights = mstWeightsVec.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    const bool useAlpha = (alpha != 1.0);

    using FP        = algorithmFPType;
    using Eucl      = EuclideanDist<FP>;
    using Manh      = ManhattanDist<FP>;
    using Mink      = MinkowskiDist<FP>;
    using Cheb      = ChebyshevDist<FP>;
    using AlphaManh = AlphaScaledDist<FP, Manh>;
    using AlphaMink = AlphaScaledDist<FP, Mink>;
    using AlphaCheb = AlphaScaledDist<FP, Cheb>;
    using AlphaEucl = AlphaScaledDist<FP, Eucl>;

    using algorithms::internal::PairwiseDistanceType;

    switch (pairwiseDistance)
    {
    case PairwiseDistanceType::manhattan:
        if (useAlpha)
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, AlphaManh(Manh(), alpha));
        else
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, Manh());
        break;
    case PairwiseDistanceType::minkowski:
        if (useAlpha)
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, AlphaMink(Mink(minkowskiDegree), alpha));
        else
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, Mink(minkowskiDegree));
        break;
    case PairwiseDistanceType::chebyshev:
        if (useAlpha)
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, AlphaCheb(Cheb(), alpha));
        else
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, Cheb());
        break;
    default: // euclidean
        if (useAlpha)
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, AlphaEucl(Eucl(), alpha));
        else
            runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                            mstTo, mstWeights, Eucl());
        break;
    }

    // Steps 4-5: Sort MST + Extract clusters (shared with brute_force/kd_tree)
    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                       clusterSelection, allowSingleCluster, clusterSelectionEpsilon, maxClusterSize);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
