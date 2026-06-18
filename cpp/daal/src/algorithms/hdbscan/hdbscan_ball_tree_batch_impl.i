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
 * Ball trees can be preferable to kd-trees for some higher-dimensional datasets,
 * although pruning efficiency still depends on the data distribution and metric.
 * The lower bound for a query point q to a ball (center c, radius r) is:
 *   max(0, dist(q, c) - r)
 */

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

/// Ball-tree node: hypersphere over a contiguous range of point indices.
///
/// Each node stores a center pivot id and a radius. The lower bound for a
/// query `q` to any point in the ball is `max(0, dist(q, center) - radius)`,
/// which is what makes ball trees more robust than kd-trees in high dimensions.
///
/// @tparam algorithmFPType Floating-point type
template <typename algorithmFPType>
struct BallNode
{
    int left;               ///< Index of left child node (-1 for leaf)
    int right;              ///< Index of right child node (-1 for leaf)
    int pointBegin;         ///< Begin of the node's point-index range
    int pointEnd;           ///< End (exclusive) of the node's point-index range
    int centerIdx;          ///< Index of the pivot point used as approximate center
    algorithmFPType radius; ///< Max distance from center to any point in this ball
    int componentId;        ///< -1 = mixed components, >= 0 = uniform Boruvka component
};

/// Gather `pointIndices[begin..end)` rows from `data` into a contiguous
/// row-major buffer.
///
/// Paying the gather cost once lets pivot2 / pivot3 / radius / partition share
/// one layout, avoids repeated FP conversions on low-precision input, and is
/// what makes the BLAS xxgemv path in EuclideanDist::blockDist applicable
/// (xxgemv needs the operand row block to be a contiguous matrix).
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  data         Row-major input buffer of size `nRows × nCols`
/// @param[in]  pointIndices Permutation array; rows `[begin, end)` are gathered
/// @param[in]  begin        First index (inclusive) into `pointIndices`
/// @param[in]  end          Last index (exclusive) into `pointIndices`
/// @param[in]  nCols        Number of features
/// @param[out] scratchRows  Output, row-major `(end - begin) × nCols`
template <typename algorithmFPType, CpuType cpu>
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

/// Index of the largest entry in `[0, count)`; tiebreak by first occurrence.
///
/// Split out from blockDistsAndArgmax so the distance pass goes entirely
/// through `distFunc.blockDist` (BLAS xxgemv on Euclidean) and the argmax
/// becomes a straight-line scalar reduction over a contiguous array.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in] arr   Input array, length `count`
/// @param[in] count Number of entries (must be >= 1)
///
/// @return Index of the maximum element
template <typename algorithmFPType, CpuType cpu>
static int argmaxArray(const algorithmFPType * arr, int count)
{
    DAAL_ASSERT(count > 0);
    algorithmFPType best = arr[0];
    int idx              = 0;
    for (int i = 1; i < count; i++)
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
/// Delegates the distance sweep to `distFunc.blockDist`. For EuclideanDist
/// this is one BLAS xxgemv + vSqrt finalize over the contiguous scratch
/// buffer; row norms^2 come from `rowNorms2` (pre-cached once per node by the
/// caller). For other metrics blockDist falls back to a vectorized per-row
/// inner loop. The argmax is a separate scalar reduction over the resulting
/// array.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor exposing `blockDist`
///
/// @param[in]  pivotPt     Pivot row, length `nCols`
/// @param[in]  scratchRows Row-major batch, size `count × nCols`
/// @param[in]  rowNorms2   Per-row squared norms, length `count` (may be unused for non-Euclidean)
/// @param[in]  count       Number of rows in the batch
/// @param[in]  nCols       Number of features
/// @param[in]  distFunc    Metric functor instance
/// @param[out] outDists    Output distances, length `count`
///
/// @return Index of the row with the maximum distance
template <typename algorithmFPType, daal::internal::CpuType cpu, typename DistFunc>
static int blockDistsAndArgmax(const algorithmFPType * pivotPt, const algorithmFPType * scratchRows, const algorithmFPType * rowNorms2, int count,
                               int nCols, const DistFunc & distFunc, algorithmFPType * outDists)
{
    distFunc.template blockDist<cpu>(pivotPt, scratchRows, rowNorms2, count, nCols, outDists);
    return argmaxArray<algorithmFPType, cpu>(outDists, count);
}

/// Recursively build a ball-tree node for `pointIndices[begin..end)`.
///
/// Each call:
///   1. Claims a unique slot `nodeIdx` from the shared `nextNode` counter
///      (deterministic: left-to-right in DFS order).
///   2. Gathers the relevant rows into a contiguous scratch buffer once so all
///      subsequent distance sweeps read contiguous memory and (for Euclidean)
///      can be batched as a single BLAS xxgemv.
///   3. Computes `‖x_i‖²` once into `rowNorms2` and reuses it across the three
///      pivot sweeps (radius is free once `d2` is filled).
///   4. Picks pivot2 as the farthest point from pivot1 (= first point), pivot3
///      as the farthest from pivot2. pivot2 becomes the ball center; `max(d2)`
///      becomes the ball radius.
///   5. Partitions points by `d2 <= d3` vs `d2 > d3`. `d2`, `d3`, and
///      `pointIndices` swap in lockstep during the partition.
///   6. Recurses sequentially: left child first, then right. Sequential
///      ordering gives reproducible node-array layout across runs.
///
/// Build cost is small relative to Boruvka MST + labeling, so sibling subtrees
/// are not parallelized here.
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
static int buildBallTree(const algorithmFPType * data, int * pointIndices, int begin, int end, int nCols, BallNode<algorithmFPType> * nodes,
                         int & nextNode, int maxLeafSize, const DistFunc & distFunc)
{
    const int nodeIdx                = nextNode++;
    BallNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                  = begin;
    node.pointEnd                    = end;
    node.componentId                 = -1;
    node.left                        = -1;
    node.right                       = -1;

    const int count = end - begin;

    // Gather rows once; all subsequent distance sweeps read contiguous memory.
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> scratchRowsArr(static_cast<size_t>(count) * nCols);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> rowNorms2Arr(count);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d2Arr(count);
    daal::services::internal::TArrayScalable<algorithmFPType, cpu> d3Arr(count);
    algorithmFPType * scratchRows = scratchRowsArr.get();
    algorithmFPType * rowNorms2   = rowNorms2Arr.get();
    algorithmFPType * d2          = d2Arr.get();
    algorithmFPType * d3          = d3Arr.get();
    if (!scratchRows || !rowNorms2 || !d2 || !d3) return nodeIdx;

    gatherRows<algorithmFPType, cpu>(data, pointIndices, begin, end, nCols, scratchRows);
    // Cache ‖x_i‖² once per node; reused by all three pivot sweeps when DistFunc is Euclidean.
    rowNormsSquared<algorithmFPType, cpu>(scratchRows, count, nCols, rowNorms2);

    // Pick pivot1 = first point. Find pivot2 = argmax dist(pivot1, ·); pos is the
    // offset into scratchRows / pointIndices[begin..end). d3 is scratch for this
    // sweep; it gets overwritten with dist-to-pivot3 below.
    const int pivot1 = pointIndices[begin];
    const int pos2   = blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot1 * nCols, scratchRows, rowNorms2, count, nCols, distFunc, d3);
    const int pivot2 = pointIndices[begin + pos2];

    // pivot2 is the ball center; d2[i] = dist(pivot2, scratchRows[i]) feeds both
    // the radius (max d2) and the partition (compared to d3).
    const int pos3   = blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot2 * nCols, scratchRows, rowNorms2, count, nCols, distFunc, d2);
    const int pivot3 = pointIndices[begin + pos3];

    node.centerIdx = pivot2;

    algorithmFPType maxR = algorithmFPType(0);
    for (int i = 0; i < count; i++)
    {
        if (d2[i] > maxR) maxR = d2[i];
    }
    node.radius = maxR;

    if (count <= maxLeafSize || count <= 1)
    {
        return nodeIdx;
    }

    // Populate d3 (distances to pivot3), then partition.
    blockDistsAndArgmax<algorithmFPType, cpu>(data + pivot3 * nCols, scratchRows, rowNorms2, count, nCols, distFunc, d3);

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
/// Maintains a bounded max-heap of the k nearest candidates seen so far. At
/// each internal node, the lower bound `max(0, dist(query, ballCenter) -
/// ballRadius)` is compared against the current k-th NN distance
/// (`heap.maxDist()`). The closer child is visited first so the pruning
/// radius tightens before exploring the farther child. At leaves the loop
/// scans every point and pushes into the heap.
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
static void knnQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                             const algorithmFPType * queryPoint, int nodeIdx, KnnHeap<algorithmFPType, cpu> & heap, const DistFunc & distFunc)
{
    const BallNode<algorithmFPType> & node = nodes[nodeIdx];

    // Prune: if the closest point in the ball is farther than current k-th NN
    const algorithmFPType distToCenter = distFunc.template pointDist<cpu>(queryPoint, data + node.centerIdx * nCols, nCols);
    const algorithmFPType lowerBound   = (distToCenter > node.radius) ? (distToCenter - node.radius) : algorithmFPType(0);
    if (lowerBound >= heap.maxDist()) return;

    if (node.left < 0)
    {
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi               = pointIndices[i];
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
/// For a query point `q` with core distance `c_q`, any candidate `p` in a
/// subtree `S` has `MRD(q, p) >= max(c_q, minCoreDistNode[S], dist(q, S))`,
/// so this aggregate is the third pruning ingredient for Boruvka MRD queries.
/// Recursion returns the subtree min so each node sees its descendants'
/// aggregate without a second pass.
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
    const algorithmFPType leftMin = computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.left);
    const algorithmFPType rightMin =
        computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.right);
    minCoreDistNode[nodeIdx] = (leftMin < rightMin) ? leftMin : rightMin;
    return minCoreDistNode[nodeIdx];
}

/// Refresh per-node component tags after a Boruvka merge round.
///
/// A node's `componentId` is set to the shared component id when every point
/// under it belongs to the same component, otherwise -1 ("mixed"). The
/// Boruvka nearest-different-component query uses this to skip whole subtrees
/// that match the query's component.
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
    const int leftComp  = updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.left);
    const int rightComp = updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.right);
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
/// Prunes subtrees whose `componentId` matches the query's component, or whose
/// MRD lower bound `max(coreDist(q), minCoreDistNode[S], max(0, dist(q,
/// center) - radius) * invAlpha)` is not smaller than the current `bestMrd`.
/// Visits the nearer child first to tighten `bestMrd` before exploring the
/// other side.
///
/// Alpha is applied only to the dist(q,p) term inside MRD (canonical HDBSCAN
/// robust single linkage); core distances are left unscaled.
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
static void nearestMrdBoruvkaQueryBallTree(const algorithmFPType * data, int nCols, const BallNode<algorithmFPType> * nodes, const int * pointIndices,
                                           const algorithmFPType * coreDistances, const algorithmFPType * minCoreDistNode, const int * componentOf,
                                           const algorithmFPType * queryPoint, int queryIdx, algorithmFPType queryCoreD, int queryComponent,
                                           int nodeIdx, algorithmFPType & bestMrd, int & bestIdx, const DistFunc & distFunc, algorithmFPType invAlpha)
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
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi = pointIndices[i];
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

    const int nearChild = (dLeft <= dRight) ? node.left : node.right;
    const int farChild  = (dLeft <= dRight) ? node.right : node.left;

    nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint,
                                                         queryIdx, queryCoreD, queryComponent, nearChild, bestMrd, bestIdx, distFunc, invAlpha);
    nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf, queryPoint,
                                                         queryIdx, queryCoreD, queryComponent, farChild, bestMrd, bestIdx, distFunc, invAlpha);
}

/// Compute core distances and the MST under MRD on a ball tree.
///
/// Pipeline:
///   1. per-point k-NN query against the ball tree -> `coreDistances`;
///   2. bottom-up reduction -> `minCoreDistNode`;
///   3. Boruvka rounds: per-point nearest-different-component MRD query,
///      reduce to per-component best edges, union via union-find, refresh
///      per-node component tags. Loops until a single component remains or no
///      progress is made. The MST has exactly `nRows - 1` edges.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor
///
/// @param[in]     data           Row-major input buffer of size `nRows × nCols`
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
///                               only to dist(q,p) inside MRD (not to k-NN
///                               core distances or to the metric used for tree
///                               queries)
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void computeCoreDistAndMstBallTree(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples,
                                          BallNode<algorithmFPType> * nodes, int * pointIndices, int totalTreeNodes, algorithmFPType * coreDistances,
                                          int * mstFrom, int * mstTo, algorithmFPType * mstWeights, const DistFunc & distFunc, double alpha)
{
    const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
    // Canonical HDBSCAN core distance (Campello 2013): the distance to the
    // `minSamples`-th nearest neighbor counting the query point itself as
    // neighbor #1. The ball-tree traversal pushes the query point into the
    // heap along with the other leaf points, so a heap of size `minSamples`
    // holds {self + (minSamples - 1) non-self}, and the heap top is the
    // `minSamples`-th-including-self answer.
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
    computeMinCoreDistsBallTree<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

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

    updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);

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

            nearestMrdBoruvkaQueryBallTree<algorithmFPType, cpu>(data, iNCols, nodes, pointIndices, coreDistances, minCoreDistNode, componentOf,
                                                                 queryPt, static_cast<int>(i), queryCoreD, comp, 0, bestMrd, bestIdx, distFunc,
                                                                 invAlpha);

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

        updateNodeComponentsBallTree<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);
    }
}

/// Build the ball tree then compute core distances + Boruvka MST under MRD.
///
/// Single entry point used by the per-metric switch in compute(): each
/// pairwise-distance value instantiates this template once. Alpha is applied
/// only to dist(q,p) inside MRD, not to the metric used for the build / k-NN. Sequential build
/// keeps the node-array layout deterministic across runs.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor
///
/// @param[in]     data          Row-major input buffer of size `nRows × nCols`
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
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void runBallTreeCoreDistAndMst(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples, int maxLeafSize,
                                      BallNode<algorithmFPType> * nodes, int * pointIndices, algorithmFPType * coreDistances, int * mstFrom,
                                      int * mstTo, algorithmFPType * mstWeights, const DistFunc & distFunc, double alpha)
{
    int nextNode = 0;
    buildBallTree<algorithmFPType, cpu>(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodes, nextNode, maxLeafSize,
                                        distFunc);
    const int totalTreeNodes = nextNode;
    computeCoreDistAndMstBallTree<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, coreDistances, mstFrom,
                                                        mstTo, mstWeights, distFunc, alpha);
}

/// Compute HDBSCAN clustering using the ball-tree based batch implementation.
///
/// Pipeline: build ball tree -> per-point k-NN core distances -> Boruvka MST
/// under MRD -> sort + extract clusters via the shared cluster-utils path.
///
/// @tparam algorithmFPType Floating-point type used for distances and lambdas
/// @tparam method          DAAL Method tag (`ballTree`)
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  ntData                  Input numeric table of size `N x P`
/// @param[out] ntAssignments           Output `N x 1` table; `-1` is noise, non-negative
///                                     values are cluster ids in `[0, C)`
/// @param[out] ntNClusters             Output `1 x 1` table holding the cluster count `C`
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
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           algorithms::internal::PairwiseDistanceType pairwiseDistance,
                                                                           double minkowskiDegree, int clusterSelection, bool allowSingleCluster,
                                                                           double clusterSelectionEpsilon, size_t maxClusterSize, double alpha,
                                                                           size_t leafSize)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();

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

    const size_t edgeCount = nRows - 1;

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

    using FP   = algorithmFPType;
    using Eucl = EuclideanDist<FP>;
    using Manh = ManhattanDist<FP>;
    using Mink = MinkowskiDist<FP>;
    using Cheb = ChebyshevDist<FP>;

    using algorithms::internal::PairwiseDistanceType;

    // Robust single linkage: alpha is applied only to dist(q,p) inside MRD
    // (canonical HDBSCAN). The metric used for ball-tree build, k-NN core
    // distances, and pruning is left unscaled.
    switch (pairwiseDistance)
    {
    case PairwiseDistanceType::manhattan:
        runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                        mstTo, mstWeights, Manh(), alpha);
        break;
    case PairwiseDistanceType::minkowski:
        runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                        mstTo, mstWeights, Mink(minkowskiDegree), alpha);
        break;
    case PairwiseDistanceType::chebyshev:
        runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                        mstTo, mstWeights, Cheb(), alpha);
        break;
    default: // euclidean
        runBallTreeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, maxLeafSize, nodes, pointIndices, coreDistances, mstFrom,
                                                        mstTo, mstWeights, Eucl(), alpha);
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
