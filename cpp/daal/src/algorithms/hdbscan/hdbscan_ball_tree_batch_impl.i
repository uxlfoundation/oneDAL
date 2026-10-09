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
 * HDBSCAN implementation using a ball tree with Boruvka's Minimum Spanning
 * Tree (MST) algorithm. Follows McInnes & Healy, "Accelerated Hierarchical
 * Density Based Clustering", https://arxiv.org/abs/1705.07321.
 *
 *   1. Build a ball tree
 *   2. Core distances from k-NN queries on the tree
 *   3. Boruvka MST under MRD with tree-pruned nearest-other-component queries
 *   4. Sort the MST and extract clusters (shared with brute force)
 *
 * The distance lower bound from a point q to a ball (center c, radius r) is max(0, dist(q, c) - r).
 */

#include <cstdint>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_boruvka_utils.h"
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

/// Ball-tree node: hypersphere over a contiguous range of point indices.
///
/// @tparam algorithmFPType Floating-point type
template <typename algorithmFPType>
struct BallNode
{
    DAAL_INT left;          ///< Index of left child node (-1 for leaf)
    DAAL_INT right;         ///< Index of right child node (-1 for leaf)
    DAAL_INT pointBegin;    ///< Begin of the node's point-index range
    DAAL_INT pointEnd;      ///< End (exclusive) of the node's point-index range
    DAAL_INT centerIdx;     ///< Index of the pivot point used as approximate center
    algorithmFPType radius; ///< Max distance from center to any point in this ball
    DAAL_INT componentId;   ///< -1 = mixed components, >= 0 = uniform Boruvka component
};

/// Gather `pointIndices[begin..end)` rows from `data` into a row-major buffer
/// with per-row padding.
///
/// `rowStride >= nCols` keeps every row aligned like the base pointer; padding cells are zero.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  data         Row-major input buffer of size `nRows x nCols`
/// @param[in]  pointIndices Permutation array; rows `[begin, end)` are gathered
/// @param[in]  begin        First index (inclusive) into `pointIndices`
/// @param[in]  end          Last index (exclusive) into `pointIndices`
/// @param[in]  nCols        Number of features
/// @param[in]  rowStride    Row stride of `scratchRows` in elements (`>= nCols`)
/// @param[out] scratchRows  Output, row-major `(end - begin) x rowStride`
template <typename algorithmFPType, CpuType cpu>
static void gatherRows(const algorithmFPType * data, const DAAL_INT * pointIndices, DAAL_INT begin, DAAL_INT end, size_t nCols, size_t rowStride,
                       algorithmFPType * scratchRows)
{
    const DAAL_INT count      = end - begin;
    const size_t rowBytes     = nCols * sizeof(algorithmFPType);
    const size_t paddingBytes = (rowStride - nCols) * sizeof(algorithmFPType);
    const size_t strideBytes  = rowStride * sizeof(algorithmFPType);
    for (DAAL_INT i = 0; i < count; i++)
    {
        const algorithmFPType * src = data + pointIndices[begin + i] * nCols;
        algorithmFPType * dst       = scratchRows + i * rowStride;
        daal::services::internal::daal_memcpy_s(dst, strideBytes, src, rowBytes);
        if (paddingBytes > 0)
        {
            services::internal::service_memset_seq<algorithmFPType, cpu>(dst + nCols, algorithmFPType(0), rowStride - nCols);
        }
    }
}

/// Index of the largest entry in `[0, count)`; tiebreak by first occurrence.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in] arr   Input array, length `count`
/// @param[in] count Number of entries (must be >= 1)
///
/// @return Index of the maximum element
template <typename algorithmFPType, CpuType cpu>
static DAAL_INT argmaxArray(const algorithmFPType * arr, DAAL_INT count)
{
    DAAL_ASSERT(count > 0);
    algorithmFPType best = arr[0];
    DAAL_INT idx         = 0;
    for (DAAL_INT i = 1; i < count; i++)
    {
        if (arr[i] > best)
        {
            best = arr[i];
            idx  = i;
        }
    }
    return idx;
}

/// Compute distances from `pivotPt` to a contiguous row block and return the argmax.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor exposing `blockDist`
///
/// @param[in]  pivotPt     Pivot row, length `nCols`
/// @param[in]  scratchRows Row-major batch, size `count x rowStride`
/// @param[in]  rowNorms2   Per-row squared norms, length `count` (may be unused for non-Euclidean)
/// @param[in]  count       Number of rows in the batch
/// @param[in]  nCols       Number of features
/// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`)
/// @param[in]  distFunc    Metric functor instance
/// @param[out] outDists    Output distances, length `count`
///
/// @return Index of the row with the maximum distance
template <typename algorithmFPType, daal::internal::CpuType cpu, typename DistFunc>
static DAAL_INT blockDistsAndArgmax(const algorithmFPType * pivotPt, const algorithmFPType * scratchRows, const algorithmFPType * rowNorms2,
                                    DAAL_INT count, size_t nCols, size_t rowStride, const DistFunc & distFunc, algorithmFPType * outDists)
{
    distFunc.template blockDist<cpu>(pivotPt, scratchRows, rowNorms2, count, nCols, rowStride, outDists);
    return argmaxArray<algorithmFPType, cpu>(outDists, count);
}

/// Recursively build a ball-tree node for `pointIndices[begin..end)`.
///
/// The center is the point farthest from the first point, the radius its largest distance, and
/// points closer to the center than to the point farthest from it go left. Subtrees are built
/// sequentially, so the node layout is the same on every run.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag (selects the scratch allocator)
/// @tparam DistFunc        Metric functor exposing `blockDist`
///
/// @param[in]     data         Row-major input buffer
/// @param[in,out] pointIndices Permutation; reordered in-place by the partition
/// @param[in]     begin        First index of the current subtree's range
/// @param[in]     end          One past the last index of the range
/// @param[in]     nCols        Number of features
/// @param[in,out] nodes        Output node array (this call writes node `nextNode`)
/// @param[in,out] nextNode     Counter of allocated nodes; post-incremented per call
/// @param[in]     maxLeafSize  Leaf cutoff
/// @param[in]     distFunc     Metric functor instance
///
/// @return Index of the node created by this call
template <typename algorithmFPType, daal::internal::CpuType cpu, typename DistFunc>
static DAAL_INT buildBallTree(const algorithmFPType * data, DAAL_INT * pointIndices, DAAL_INT begin, DAAL_INT end, size_t nCols,
                              BallNode<algorithmFPType> * nodes, DAAL_INT & nextNode, DAAL_INT maxLeafSize, const DistFunc & distFunc)
{
    const DAAL_INT nodeIdx           = nextNode++;
    BallNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                  = begin;
    node.pointEnd                    = end;
    node.componentId                 = -1;
    node.left                        = -1;
    node.right                       = -1;

    const DAAL_INT count = end - begin;

    const size_t rowStride = alignedRowStride<algorithmFPType>(nCols);

    daal::services::internal::TArrayScalable<algorithmFPType, cpu> scratchRowsArr(static_cast<size_t>(count) * rowStride);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> rowNorms2Arr(count);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d2Arr(count);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d3Arr(count);
    algorithmFPType * scratchRows = scratchRowsArr.get();
    algorithmFPType * rowNorms2   = rowNorms2Arr.get();
    algorithmFPType * d2          = d2Arr.get();
    algorithmFPType * d3          = d3Arr.get();
    if (!scratchRows || !rowNorms2 || !d2 || !d3) return nodeIdx;

    gatherRows<algorithmFPType, cpu>(data, pointIndices, begin, end, nCols, rowStride, scratchRows);
    // Cache ||x_i||^2 once per node; reused by all three pivot sweeps when DistFunc is Euclidean.
    rowNormsSquared<algorithmFPType, cpu>(scratchRows, count, nCols, rowStride, rowNorms2);

    // pivot2 is the point farthest from the first point; d3 is scratch here.
    const DAAL_INT pivot1 = pointIndices[begin];
    const DAAL_INT pos2 =
        blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot1 * nCols, scratchRows, rowNorms2, count, nCols, rowStride, distFunc, d3);
    const DAAL_INT pivot2 = pointIndices[begin + pos2];

    // pivot2 is the center; d2 gives the radius and, against d3, the partition.
    const DAAL_INT pos3 =
        blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot2 * nCols, scratchRows, rowNorms2, count, nCols, rowStride, distFunc, d2);
    const DAAL_INT pivot3 = pointIndices[begin + pos3];

    node.centerIdx = pivot2;

    algorithmFPType maxR = algorithmFPType(0);
    // d2 comes from TArrayScalable, aligned to DAAL_MALLOC_DEFAULT_ALIGNMENT.
    PRAGMA_OMP_SIMD_ARGS(reduction(max : maxR) aligned(d2 : DAAL_MALLOC_DEFAULT_ALIGNMENT))
    for (DAAL_INT i = 0; i < count; i++)
    {
        maxR = (d2[i] > maxR) ? d2[i] : maxR;
    }
    node.radius = maxR;

    if (count <= maxLeafSize || count <= 1)
    {
        return nodeIdx;
    }

    // Populate d3 (distances to pivot3), then partition.
    blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot3 * nCols, scratchRows, rowNorms2, count, nCols, rowStride, distFunc, d3);

    // Hoare partition on `d2[i] <= d3[i]`; pointIndices, d2 and d3 swap together.
    DAAL_INT lo = 0;
    DAAL_INT hi = count - 1;
    while (lo <= hi)
    {
        if (d2[lo] <= d3[lo])
        {
            lo++;
        }
        else
        {
            services::internal::swap<cpu>(pointIndices[begin + lo], pointIndices[begin + hi]);
            services::internal::swap<cpu>(d2[lo], d2[hi]);
            services::internal::swap<cpu>(d3[lo], d3[hi]);
            hi--;
        }
    }
    DAAL_INT mid = begin + lo;

    // Ensure both sides are non-empty
    if (mid == begin || mid == end)
    {
        mid = begin + count / 2;
    }

    node.left  = buildBallTree<algorithmFPType, cpu>(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, distFunc);
    node.right = buildBallTree<algorithmFPType, cpu>(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, distFunc);

    return nodeIdx;
}

/// k-nearest-neighbor query on the ball tree, pruned by hypersphere bounds.
///
/// A child is skipped when its lower bound is not below the current k-th distance.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor exposing `pointDist`
///
/// @param[in]     data         Row-major input buffer
/// @param[in]     nCols        Number of features
/// @param[in]     nodes        Ball-tree nodes
/// @param[in]     pointIndices Point-index permutation owned by the tree
/// @param[in]     queryPoint   Query row, length `nCols`
/// @param[in]     nodeIdx      Subtree root to visit (caller passes 0)
/// @param[in,out] heap         Bounded max-heap of best-k candidates seen so far
/// @param[in]     distFunc     Metric functor instance
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void knnQueryBallTree(const algorithmFPType * data, size_t nCols, const BallNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices,
                             const algorithmFPType * queryPoint, DAAL_INT nodeIdx, KnnHeap<algorithmFPType, cpu> & heap, const DistFunc & distFunc)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];

    // Prune: if the closest point in the ball is farther than current k-th NN
    const algorithmFPType distToCenter = distFunc.template pointDist<cpu>(queryPoint, data + node.centerIdx * nCols, nCols);
    const algorithmFPType lowerBound   = (distToCenter > node.radius) ? (distToCenter - node.radius) : algorithmFPType(0);
    if (lowerBound >= heap.maxDist()) return;

    if (node.left < 0)
    {
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const DAAL_INT pi          = pointIndices[i];
            const algorithmFPType dist = distFunc.template pointDist<cpu>(queryPoint, data + pi * nCols, nCols);
            heap.push(dist, pi);
        }
        return;
    }

    // Visit nearer child first
    const algorithmFPType dLeft  = distFunc.template pointDist<cpu>(queryPoint, data + nodes[node.left].centerIdx * nCols, nCols);
    const algorithmFPType dRight = distFunc.template pointDist<cpu>(queryPoint, data + nodes[node.right].centerIdx * nCols, nCols);

    if (dLeft <= dRight)
    {
        knnQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, node.left, heap, distFunc);
        knnQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, node.right, heap, distFunc);
    }
    else
    {
        knnQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, node.right, heap, distFunc);
        knnQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, node.left, heap, distFunc);
    }
}

/// Compute the per-node minimum core distance bottom-up.
///
/// A lower bound for the MRD queries: `MRD(q, p) >= max(c_q, minCoreDistNode[S], dist(q, S))`.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  nodes           Ball-tree nodes
/// @param[in]  pointIndices    Point-index permutation
/// @param[in]  coreDistances   Per-point core distances, length `nRows`
/// @param[out] minCoreDistNode Per-node min core distance, length `totalTreeNodes`
/// @param[in]  nodeIdx         Subtree root (caller passes 0)
///
/// @return Min core distance found in this subtree
template <typename algorithmFPType, CpuType cpu>
static algorithmFPType computeMinCoreDistsBallTree(const BallNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices,
                                                   const algorithmFPType * coreDistances, algorithmFPType * minCoreDistNode, DAAL_INT nodeIdx)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.left < 0)
    {
        algorithmFPType minCD = daal::services::internal::MaxVal<algorithmFPType>::get();
        PRAGMA_OMP_SIMD_ARGS(reduction(min : minCD))
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const algorithmFPType cd = coreDistances[pointIndices[i]];
            minCD                    = (cd < minCD) ? cd : minCD;
        }
        minCoreDistNode[nodeIdx] = minCD;
        return minCD;
    }

    const algorithmFPType leftMin = computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.left);
    const algorithmFPType rightMin =
        computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.right);
    minCoreDistNode[nodeIdx] = (leftMin < rightMin) ? leftMin : rightMin;
    return minCoreDistNode[nodeIdx];
}

/// Refresh per-node component tags after a Boruvka merge round.
///
/// A node gets its points' common component, or -1 if mixed.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in,out] nodes        Ball-tree nodes (component ids written in place)
/// @param[in]     pointIndices Point-index permutation
/// @param[in]     componentOf  Per-point component id, length `nRows`
/// @param[in]     nodeIdx      Subtree root (caller passes 0)
///
/// @return Shared component id of this subtree, or -1 if mixed
template <typename algorithmFPType, CpuType cpu>
static DAAL_INT updateNodeComponentsBallTree(BallNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices, const DAAL_INT * componentOf,
                                             DAAL_INT nodeIdx)
{
    BallNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.left < 0)
    {
        const DAAL_INT firstComp = componentOf[pointIndices[node.pointBegin]];
        for (DAAL_INT i = node.pointBegin + 1; i < node.pointEnd; i++)
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
    const DAAL_INT leftComp  = updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.left);
    const DAAL_INT rightComp = updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.right);
    if (leftComp >= 0 && leftComp == rightComp)
    {
        node.componentId = leftComp;
        return leftComp;
    }
    node.componentId = -1;
    return -1;
}

/// Find the query's nearest point in a different component under MRD on the ball tree.
///
/// Skips subtrees whose points all share the query's component, and subtrees whose MRD lower
/// bound is not below the best so far. Alpha scales only the distance term.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor exposing `pointDist`
///
/// @param[in]     data            Row-major input buffer
/// @param[in]     nCols           Number of features
/// @param[in]     nodes           Ball-tree nodes
/// @param[in]     pointIndices    Point-index permutation
/// @param[in]     coreDistances   Per-point core distances, length `nRows`
/// @param[in]     minCoreDistNode Per-node minimum core distance
/// @param[in]     componentOf     Per-point component id
/// @param[in]     queryPoint      Query row, length `nCols`
/// @param[in]     queryIdx        Query point index (used by callers for self-skip)
/// @param[in]     queryCoreD      Query's core distance
/// @param[in]     queryComponent  Query's current component id
/// @param[in]     nodeIdx         Subtree root (caller passes 0)
/// @param[in,out] bestMrd         Best MRD found so far (caller seeds with `+inf`)
/// @param[in,out] bestIdx         Index of the best different-component candidate so far
/// @param[in]     distFunc        Metric functor instance (unscaled metric)
/// @param[in]     invAlpha        `1.0 / alpha`, applied only to dist(q,p) inside MRD
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void nearestMrdBoruvkaQueryBallTree(const algorithmFPType * data, size_t nCols, const BallNode<algorithmFPType> * nodes,
                                           const DAAL_INT * pointIndices, const algorithmFPType * coreDistances,
                                           const algorithmFPType * minCoreDistNode, const DAAL_INT * componentOf, const algorithmFPType * queryPoint,
                                           DAAL_INT queryIdx, algorithmFPType queryCoreD, DAAL_INT queryComponent, DAAL_INT nodeIdx,
                                           algorithmFPType & bestMrd, DAAL_INT & bestIdx, const DistFunc & distFunc, algorithmFPType invAlpha)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.componentId == queryComponent) return;

    // Ball-tree MRD lower bound: max(dist_to_center - radius, 0) * invAlpha then max with cores
    const algorithmFPType distToCenter = distFunc.template pointDist<cpu>(queryPoint, data + node.centerIdx * nCols, nCols);
    const algorithmFPType bboxMin      = (distToCenter > node.radius) ? (distToCenter - node.radius) : algorithmFPType(0);
    algorithmFPType mrdLB              = bboxMin * invAlpha;
    if (queryCoreD > mrdLB) mrdLB = queryCoreD;
    if (minCoreDistNode[nodeIdx] > mrdLB) mrdLB = minCoreDistNode[nodeIdx];
    if (mrdLB >= bestMrd) return;

    if (node.left < 0)
    {
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const DAAL_INT pi = pointIndices[i];
            if (componentOf[pi] == queryComponent) continue;

            const algorithmFPType dist = distFunc.template pointDist<cpu>(queryPoint, data + pi * nCols, nCols);
            algorithmFPType mrd        = dist * invAlpha;
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
    const algorithmFPType dLeft  = distFunc.template pointDist<cpu>(queryPoint, data + nodes[node.left].centerIdx * nCols, nCols);
    const algorithmFPType dRight = distFunc.template pointDist<cpu>(queryPoint, data + nodes[node.right].centerIdx * nCols, nCols);

    const DAAL_INT nearChild = (dLeft <= dRight) ? node.left : node.right;
    const DAAL_INT farChild  = (dLeft <= dRight) ? node.right : node.left;

    nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint,
                                                         queryIdx, queryCoreD, queryComponent, nearChild, bestMrd, bestIdx, distFunc, invAlpha);
    nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint,
                                                         queryIdx, queryCoreD, queryComponent, farChild, bestMrd, bestIdx, distFunc, invAlpha);
}

/// Compute core distances and the MST under MRD on a ball tree.
///
/// Core distances from k-NN queries, then Boruvka rounds until one component remains.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor
///
/// @param[in]     data           Row-major input buffer of size `nRows x nCols`
/// @param[in]     nRows          Number of points
/// @param[in]     nCols          Number of features
/// @param[in]     minSamples     k for core-distance k-NN queries
/// @param[in,out] nodes          Ball-tree nodes (component tags refreshed each round)
/// @param[in]     pointIndices   Point-index permutation produced by buildBallTree
/// @param[in]     totalTreeNodes Number of ball-tree nodes
/// @param[out]    coreDistances  Per-point core distances, length `nRows`
/// @param[out]    mstFrom        Source endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstTo          Target endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstWeights     Edge weights (MRD), length `nRows - 1`
/// @param[in]     distFunc       Metric functor instance (unscaled metric)
/// @param[in]     alpha          Robust single-linkage scaling factor; applied
///                               only to dist(q,p) inside MRD
///
/// @return Number of MST edges emitted; fewer than `nRows - 1` only on non-finite input
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static size_t computeCoreDistAndMstBallTree(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples,
                                            BallNode<algorithmFPType> * nodes, DAAL_INT * pointIndices, DAAL_INT totalTreeNodes,
                                            algorithmFPType * coreDistances, DAAL_INT * mstFrom, DAAL_INT * mstTo, algorithmFPType * mstWeights,
                                            const DistFunc & distFunc, double alpha)
{
    const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
    // The core distance is the distance to the minSamples-th nearest neighbor counting the point
    // itself, i.e. the heap top of a size-minSamples query that includes the point.
    const DAAL_INT k = static_cast<DAAL_INT>(minSamples);

    // Step 2: Core distances via k-NN on ball tree
    daal::threader_for(nRows, 1, [&](size_t i) {
        KnnHeap<algorithmFPType, cpu> heap(k);
        if (!heap.ok()) return;

        knnQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, data + i * nCols, 0, heap, distFunc);

        coreDistances[i] = heap.maxDist();
    });

    // Per-node minimum core distances
    TArrayScalable<algorithmFPType, cpu> minCoreDistNodeVec(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeVec.get();
    if (!minCoreDistNode) return 0;
    computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // Step 3: Boruvka MST
    TArray<DAAL_INT, cpu> ufParentVec(nRows);
    TArray<DAAL_INT, cpu> ufRankVec(nRows);
    TArray<DAAL_INT, cpu> componentOfVec(nRows);
    DAAL_INT * ufParent    = ufParentVec.get();
    DAAL_INT * ufRank      = ufRankVec.get();
    DAAL_INT * componentOf = componentOfVec.get();
    if (!ufParent || !ufRank || !componentOf) return 0;

    TArrayScalable<algorithmFPType, cpu> pointBestMrdVec(nRows);
    TArray<DAAL_INT, cpu> pointBestIdxVec(nRows);
    algorithmFPType * pointBestMrd = pointBestMrdVec.get();
    DAAL_INT * pointBestIdx        = pointBestIdxVec.get();
    if (!pointBestMrd || !pointBestIdx) return 0;

    TArrayScalable<algorithmFPType, cpu> compBestMrdVec(nRows);
    TArray<DAAL_INT, cpu> compBestFromVec(nRows);
    TArray<DAAL_INT, cpu> compBestToVec(nRows);
    algorithmFPType * compBestMrd = compBestMrdVec.get();
    DAAL_INT * compBestFrom       = compBestFromVec.get();
    DAAL_INT * compBestTo         = compBestToVec.get();
    if (!compBestMrd || !compBestFrom || !compBestTo) return 0;

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]    = static_cast<DAAL_INT>(i);
        componentOf[i] = static_cast<DAAL_INT>(i);
    }
    services::internal::service_memset_seq<DAAL_INT, cpu>(ufRank, 0, nRows);

    UnionFind uf { ufParent, ufRank };

    updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);

    size_t edgesAdded    = 0;
    size_t numComponents = nRows;

    // Nearest neighbor of each point outside its component, under MRD.
    while (numComponents > 1)
    {
        daal::threader_for(nRows, 1, [&](size_t i) {
            const DAAL_INT comp              = componentOf[i];
            const algorithmFPType * queryPt  = data + i * nCols;
            const algorithmFPType queryCoreD = coreDistances[i];
            algorithmFPType bestMrd          = daal::services::internal::MaxVal<algorithmFPType>::get();
            DAAL_INT bestIdx                 = -1;

            nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf,
                                                                 queryPt, static_cast<DAAL_INT>(i), queryCoreD, comp, 0, bestMrd, bestIdx, distFunc,
                                                                 invAlpha);

            pointBestMrd[i] = bestMrd;
            pointBestIdx[i] = bestIdx;
        });

        reduceComponentBestEdges<algorithmFPType>(nRows, componentOf, pointBestMrd, pointBestIdx, compBestMrd, compBestFrom, compBestTo);

        const size_t addedThisRound = mergeComponentsEmitEdges<algorithmFPType>(nRows, compBestMrd, compBestFrom, compBestTo, uf, mstFrom, mstTo,
                                                                                mstWeights, edgesAdded, numComponents);

        if (addedThisRound == 0) break;

        refreshComponentIds<cpu>(nRows, uf, componentOf);

        updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);
    }
    return edgesAdded;
}

/// Build the ball tree then compute core distances + Boruvka MST under MRD.
///
/// Alpha scales only dist(q, p) inside MRD.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor
///
/// @param[in]     data          Row-major input buffer of size `nRows x nCols`
/// @param[in]     nRows         Number of points
/// @param[in]     nCols         Number of features
/// @param[in]     minSamples    k for core-distance k-NN queries
/// @param[in]     maxLeafSize   Leaf cutoff passed to buildBallTree
/// @param[out]    nodes         Ball-tree node storage (caller pre-allocates)
/// @param[in,out] pointIndices  Point-index permutation, seeded `[0, nRows)` by caller
/// @param[out]    coreDistances Per-point core distances, length `nRows`
/// @param[out]    mstFrom       Source endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstTo         Target endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstWeights    Edge weights (MRD), length `nRows - 1`
/// @param[in]     distFunc      Metric functor instance (unscaled metric)
/// @param[in]     alpha         Robust single-linkage scaling factor; applied
///                              only to dist(q,p) inside MRD
///
/// @return Number of MST edges emitted, see computeCoreDistAndMstBallTree
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static size_t runBallTreeCoreDistAndMst(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples, DAAL_INT maxLeafSize,
                                        BallNode<algorithmFPType> * nodes, DAAL_INT * pointIndices, algorithmFPType * coreDistances,
                                        DAAL_INT * mstFrom, DAAL_INT * mstTo, algorithmFPType * mstWeights, const DistFunc & distFunc, double alpha)
{
    DAAL_INT nextNode = 0;
    buildBallTree<algorithmFPType, cpu>(data, pointIndices, 0, static_cast<DAAL_INT>(nRows), nCols, nodes, nextNode, maxLeafSize, distFunc);
    const DAAL_INT totalTreeNodes = nextNode;
    return computeCoreDistAndMstBallTree<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, coreDistances,
                                                               mstFrom, mstTo, mstWeights, distFunc, alpha);
}

/// Compute HDBSCAN clustering using the ball-tree based batch implementation.
///
/// @tparam algorithmFPType Floating-point type used for distances and lambdas
/// @tparam method          DAAL Method tag (`ballTree`)
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  ntData                  Input numeric table of size `N x P`
/// @param[out] ntAssignments           Output `N x 1` table; `-1` is noise, non-negative
///                                     values are cluster ids in `[0, C)`
/// @param[out] ntNClusters             Output `1 x 1` table holding the cluster count `C`
/// @param[out] ntProbabilities         Optional output `N x 1` table holding the membership strength of
///                                     each point in `[0, 1]`; `nullptr` skips the computation
/// @param[out] ntSingleLinkageTree     Optional output `(N - 1) x 4` table holding the single-linkage
///                                     dendrogram, `[left, right, distance, size]` per merge;
///                                     `nullptr` skips it
/// @param[in]  minClusterSize          Minimum cluster size threshold (mcs)
/// @param[in]  minSamples              Number of neighbors used for core distances (k)
/// @param[in]  pairwiseDistance        Distance metric tag
/// @param[in]  minkowskiDegree         Minkowski exponent (used only when metric is minkowski)
/// @param[in]  clusterSelection        0 = Excess of Mass, 1 = leaf
/// @param[in]  allowSingleCluster      If false, reject root-only EOM outcomes
/// @param[in]  clusterSelectionEpsilon Distance threshold for the cluster-epsilon merge pass
/// @param[in]  maxClusterSize          Maximum allowed cluster size; 0 disables the cap
/// @param[in]  alpha                   Robust single-linkage scaling factor
/// @param[in]  leafSize                Maximum points per ball-tree leaf
///
/// @return Status code
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable * ntData, NumericTable * ntAssignments, NumericTable * ntNClusters, NumericTable * ntProbabilities,
    NumericTable * ntSingleLinkageTree, size_t minClusterSize, size_t minSamples, algorithms::internal::PairwiseDistanceType pairwiseDistance,
    double minkowskiDegree, int clusterSelection, bool allowSingleCluster, double clusterSelectionEpsilon, size_t maxClusterSize, double alpha,
    size_t leafSize)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();

    if (nRows < 2 || minClusterSize < 2)
    {
        WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(assignBlock);
        int * assignments = assignBlock.get();
        services::internal::service_memset<int, cpu>(assignments, -1, nRows);

        WriteOnlyRows<algorithmFPType, cpu> probBlock;
        algorithmFPType * probabilities = probBlock.set(ntProbabilities, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(probBlock);
        if (probabilities)
        {
            services::internal::service_memset<algorithmFPType, cpu>(probabilities, algorithmFPType(0), nRows);
        }

        WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(ncBlock);
        ncBlock.get()[0] = 0;
        return services::Status();
    }

    // Labels are stored as int32, and there are at most nRows / minClusterSize clusters.
    if (nRows / minClusterSize > static_cast<size_t>(INT32_MAX))
    {
        return services::Status(services::ErrorIncorrectSizeOfInputNumericTable);
    }

    const size_t edgeCount = nRows - 1;

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // Step 1: Build ball tree
    const DAAL_INT maxLeafSize = static_cast<DAAL_INT>(leafSize);
    // A binary tree over nRows points has at most 2 * nRows - 1 nodes; 4 * nRows is headroom.
    const DAAL_INT maxNodes = 4 * static_cast<DAAL_INT>(nRows);

    TArray<BallNode<algorithmFPType>, cpu> nodesVec(maxNodes);
    BallNode<algorithmFPType> * nodes = nodesVec.get();
    DAAL_CHECK_MALLOC(nodes);

    TArray<DAAL_INT, cpu> pointIndicesVec(nRows);
    DAAL_INT * pointIndices = pointIndicesVec.get();
    DAAL_CHECK_MALLOC(pointIndices);
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<DAAL_INT>(i);

    TArray<algorithmFPType, cpu> coreDistsVec(nRows);
    algorithmFPType * coreDistances = coreDistsVec.get();
    DAAL_CHECK_MALLOC(coreDistances);

    TArray<DAAL_INT, cpu> mstFromVec(edgeCount);
    TArray<DAAL_INT, cpu> mstToVec(edgeCount);
    TArray<algorithmFPType, cpu> mstWeightsVec(edgeCount);
    DAAL_INT * mstFrom           = mstFromVec.get();
    DAAL_INT * mstTo             = mstToVec.get();
    algorithmFPType * mstWeights = mstWeightsVec.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    // Alpha scales only dist(q, p) inside MRD. Cosine is not an L_p distance, so the tree cannot
    // prune with it.
    size_t edgesAdded       = 0;
    services::Status status = callWithLpDistance<algorithmFPType>(pairwiseDistance, minkowskiDegree, [&](const auto & distFunc) {
        edgesAdded = runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances,
                                                                     mstFrom, mstTo, mstWeights, distFunc, alpha);
    });
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_CHECK_STATUS(status, checkMst(edgesAdded, edgeCount, mstWeights));

    // Steps 4-5: Sort MST + Extract clusters (shared with brute_force/kd_tree)
    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> probBlock;
    algorithmFPType * probabilities = probBlock.set(ntProbabilities, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(probBlock);

    WriteOnlyRows<algorithmFPType, cpu> sltBlock;
    algorithmFPType * singleLinkageTree = sltBlock.set(ntSingleLinkageTree, 0, edgeCount);
    DAAL_CHECK_BLOCK_STATUS(sltBlock);

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                       clusterSelection, allowSingleCluster, clusterSelectionEpsilon, maxClusterSize,
                                                                       probabilities, singleLinkageTree);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
