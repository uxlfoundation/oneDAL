/* file: hdbscan_kd_tree_batch_impl.i */
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
 * HDBSCAN implementation using a k-d tree with Boruvka's Minimum Spanning
 * Tree (MST) algorithm. Follows McInnes & Healy, "Accelerated Hierarchical
 * Density Based Clustering", https://arxiv.org/abs/1705.07321.
 *
 * The approach:
 *   1. Build a k-d tree over the input data (with bounding boxes per node)
 *   2. For each point, find its k-th nearest neighbor via tree search -> core distances
 *   3. Build MST under Mutual Reachability Distance using Boruvka's algorithm
 *      with kd-tree-accelerated nearest-different-component queries
 *   4. Sort MST + extract clusters via condensed tree + EOM (shared code)
 *
 * Key advantage over brute_force: O(N * k * log N) for core distances,
 * O(N * log^2 N) for MST via Boruvka (vs O(N^2) Prim's in brute_force).
 * Memory: O(N * D * tree_nodes) for bounding boxes + O(N) working arrays.
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

/// k-d tree node.
///
/// Internal nodes split a contiguous range of `pointIndices` along `splitDim`
/// at `splitVal`; leaves are marked with `splitDim < 0`. `componentId` is
/// updated as Boruvka rounds merge components and is used to prune subtrees
/// whose points all belong to the query's current component.
///
/// Range endpoints (`pointBegin`, `pointEnd`), node indices (`left`, `right`),
/// and the component id are stored as `DAAL_INT` so trees over > INT32_MAX
/// points keep addressing addressed correctly. `splitDim` stays `int` because
/// column count is small.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct KdNode
{
    int splitDim;         ///< Splitting dimension (-1 for leaf)
    FPType splitVal;      ///< Split value along `splitDim` (unused for leaves)
    DAAL_INT left;        ///< Index of left child node (-1 if leaf)
    DAAL_INT right;       ///< Index of right child node (-1 if leaf)
    DAAL_INT pointBegin;  ///< Begin of the node's point-index range
    DAAL_INT pointEnd;    ///< End (exclusive) of the node's point-index range
    DAAL_INT componentId; ///< -1 = mixed components, >= 0 = all points in same component
};

/// Build a k-d tree recursively with per-node axis-aligned bounding boxes.
///
/// Splits along the dimension with the largest spread, partitioning
/// `pointIndices[begin..end)` around the median value via `std::nth_element`.
/// Leaves are emitted when the point count drops to `<= maxLeafSize`.
/// Bounding-box arrays are filled in place for every node and used later for
/// kd-tree pruning.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     data         Row-major input buffer of size `nRows x nCols`
/// @param[in,out] pointIndices Permutation of input row ids; reordered in place
/// @param[in]     begin        First index of the current subtree's range
/// @param[in]     end          One past the last index of the current range
/// @param[in]     nCols        Number of features
/// @param[out]    nodes        Output node array (this call writes node `nextNode`)
/// @param[in,out] nextNode     Counter of allocated nodes (post-incremented)
/// @param[in]     maxLeafSize  Max points per leaf
/// @param[out]    bboxLo       Per-node lower bbox bounds, length `totalNodes x nCols`
/// @param[out]    bboxHi       Per-node upper bbox bounds, length `totalNodes x nCols`
///
/// @return Index of the node created by this call
template <typename algorithmFPType, CpuType cpu>
static DAAL_INT buildKdTree(const algorithmFPType * data, DAAL_INT * pointIndices, DAAL_INT begin, DAAL_INT end, size_t nCols,
                            KdNode<algorithmFPType> * nodes, DAAL_INT & nextNode, DAAL_INT maxLeafSize, algorithmFPType * bboxLo,
                            algorithmFPType * bboxHi)
{
    const DAAL_INT nodeIdx         = nextNode++;
    KdNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                = begin;
    node.pointEnd                  = end;
    node.componentId               = -1;

    // Compute bounding box and find dimension with largest spread
    algorithmFPType bestSpread = algorithmFPType(-1);
    int bestDim                = 0;
    for (size_t d = 0; d < nCols; d++)
    {
        algorithmFPType lo = daal::services::internal::MaxVal<algorithmFPType>::get();
        algorithmFPType hi = -daal::services::internal::MaxVal<algorithmFPType>::get();
        // Reduction body uses `?:` rather than `if (...) x = ...` per OMP simd conformance.
        PRAGMA_OMP_SIMD_ARGS(reduction(min : lo) reduction(max : hi))
        for (DAAL_INT i = begin; i < end; i++)
        {
            const algorithmFPType val = data[pointIndices[i] * nCols + d];
            lo                        = (val < lo) ? val : lo;
            hi                        = (val > hi) ? val : hi;
        }
        bboxLo[nodeIdx * nCols + d]  = lo;
        bboxHi[nodeIdx * nCols + d]  = hi;
        const algorithmFPType spread = hi - lo;
        if (spread > bestSpread)
        {
            bestSpread = spread;
            bestDim    = static_cast<int>(d);
        }
    }

    const DAAL_INT count = end - begin;
    if (count <= maxLeafSize)
    {
        node.splitDim = -1;
        node.splitVal = 0;
        node.left     = -1;
        node.right    = -1;
        return nodeIdx;
    }

    node.splitDim = bestDim;

    // Median-of-3 partitioning via nth_element on the split dimension
    const DAAL_INT mid = begin + count / 2;
    std::nth_element(pointIndices + begin, pointIndices + mid, pointIndices + end,
                     [&](DAAL_INT a, DAAL_INT b) { return data[a * nCols + bestDim] < data[b * nCols + bestDim]; });

    node.splitVal = data[pointIndices[mid] * nCols + bestDim];

    node.left  = buildKdTree<algorithmFPType, cpu>(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);
    node.right = buildKdTree<algorithmFPType, cpu>(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);

    return nodeIdx;
}

/// k-nearest-neighbor query on the kd-tree, templated on the distance functor.
///
/// Visits the nearer child first to tighten the heap's pruning radius, then
/// recurses into the far child only if the splitting plane is closer than the
/// current k-th nearest distance.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag (forwarded to the heap allocator)
/// @tparam DistFunc        Metric functor exposing `pointDist`
///
/// @param[in]     data         Row-major input buffer
/// @param[in]     nCols        Number of features
/// @param[in]     nodes        kd-tree nodes
/// @param[in]     pointIndices Point-index permutation owned by the tree
/// @param[in]     queryPoint   Query row, length `nCols`
/// @param[in]     nodeIdx      Index of the subtree root to visit (caller passes 0 for the root)
/// @param[in,out] heap         Bounded max-heap of best-k candidates seen so far
/// @param[in]     distFunc     Metric functor instance
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void knnQuery(const algorithmFPType * data, size_t nCols, const KdNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices,
                     const algorithmFPType * queryPoint, DAAL_INT nodeIdx, KnnHeap<algorithmFPType, cpu> & heap, const DistFunc & distFunc)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const DAAL_INT pi           = pointIndices[i];
            const algorithmFPType * row = data + pi * nCols;
            const algorithmFPType dist  = distFunc.template pointDist<cpu>(queryPoint, row, nCols);
            heap.push(dist, pi);
        }
        return;
    }

    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const DAAL_INT nearChild = (diff <= 0) ? node.left : node.right;
    const DAAL_INT farChild  = (diff <= 0) ? node.right : node.left;

    knnQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, nearChild, heap, distFunc);

    // Prune: only visit far subtree if the splitting plane is closer than the
    // current k-th NN. Every L_p metric supported here reduces to |diff| for an
    // axis-aligned split, so we inline the abs instead of routing through the
    // functor. Inlining lets the compiler fold the compare into the vectorized
    // traversal loop.
    const algorithmFPType pd = (diff < algorithmFPType(0)) ? -diff : diff;
    if (pd < heap.maxDist())
    {
        knnQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, queryPoint, farChild, heap, distFunc);
    }
}

/// Compute the minimum core distance among the points in every kd-tree subtree.
///
/// Bottom-up O(N) traversal. Used as the third pruning lower bound during
/// Boruvka MRD queries (`MRD >= max(coreQ, minCoreDistNode[subtree], minDist)`).
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  nodes           kd-tree nodes
/// @param[in]  pointIndices    Point-index permutation
/// @param[in]  coreDistances   Core distance per original point, length `nRows`
/// @param[out] minCoreDistNode Per-node min core distance, length `totalNodes`
/// @param[in]  nodeIdx         Index of the subtree root (caller passes 0 for the root)
///
/// @return Min core distance found in this subtree
template <typename algorithmFPType, CpuType cpu>
static algorithmFPType computeMinCoreDists(const KdNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices,
                                           const algorithmFPType * coreDistances, algorithmFPType * minCoreDistNode, DAAL_INT nodeIdx)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.splitDim < 0)
    {
        algorithmFPType minCD = daal::services::internal::MaxVal<algorithmFPType>::get();
        // Reduction body uses `?:` per OMP simd conformance.
        PRAGMA_OMP_SIMD_ARGS(reduction(min : minCD))
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const algorithmFPType cd = coreDistances[pointIndices[i]];
            minCD                    = (cd < minCD) ? cd : minCD;
        }
        minCoreDistNode[nodeIdx] = minCD;
        return minCD;
    }
    const algorithmFPType leftMin  = computeMinCoreDists<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.left);
    const algorithmFPType rightMin = computeMinCoreDists<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, node.right);
    minCoreDistNode[nodeIdx]       = (leftMin < rightMin) ? leftMin : rightMin;
    return minCoreDistNode[nodeIdx];
}

/// Refresh per-node component ids after a Boruvka merge round.
///
/// Bottom-up traversal: a leaf inherits its single shared component if all of
/// its points agree, otherwise it is marked mixed. Internal nodes inherit the
/// component when both children agree, mixed otherwise. Pure-component nodes
/// let later MRD queries prune entire subtrees.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in,out] nodes        kd-tree nodes (component ids written in place)
/// @param[in]     pointIndices Point-index permutation
/// @param[in]     componentOf  Per-point component id, length `nRows`
/// @param[in]     nodeIdx      Index of the subtree root (caller passes 0)
///
/// @return The shared component id of this subtree, or -1 if mixed
template <typename algorithmFPType, CpuType cpu>
static DAAL_INT updateNodeComponents(KdNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices, const DAAL_INT * componentOf, DAAL_INT nodeIdx)
{
    KdNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.splitDim < 0)
    {
        // Leaf: check if all points share the same component
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
    const DAAL_INT leftComp  = updateNodeComponents<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.left);
    const DAAL_INT rightComp = updateNodeComponents<algorithmFPType, cpu>(nodes, pointIndices, componentOf, node.right);
    if (leftComp >= 0 && leftComp == rightComp)
    {
        node.componentId = leftComp;
        return leftComp;
    }
    node.componentId = -1;
    return -1;
}

/// Find the query's nearest point in a different component under MRD on the kd-tree.
///
/// Three-level pruning:
///   1. skip subtrees whose `componentId` matches the query's component;
///   2. skip subtrees whose `max(coreQ, minCoreDistNode, bboxMinDist * invAlpha)`
///      is not smaller than the current best MRD;
///   3. visit the nearer child first to tighten `bestMrd`, then the far child
///      only if its plane distance still permits an improvement.
///
/// Alpha is applied only to the dist(q,p) term inside MRD (canonical HDBSCAN
/// robust single linkage); core distances are left unscaled.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor exposing `pointDist` and `bboxLowerBound`
///
/// @param[in]     data            Row-major input buffer
/// @param[in]     nCols           Number of features
/// @param[in]     nodes           kd-tree nodes
/// @param[in]     pointIndices    Point-index permutation
/// @param[in]     coreDistances   Per-point core distances, length `nRows`
/// @param[in]     bboxLo          Per-node lower bbox bounds
/// @param[in]     bboxHi          Per-node upper bbox bounds
/// @param[in]     minCoreDistNode Per-node minimum core distance
/// @param[in]     componentOf     Per-point component id
/// @param[in]     queryPoint      Query row, length `nCols`
/// @param[in]     queryIdx        Query point index (used only for self-skip in callers)
/// @param[in]     queryCoreD      Query's core distance
/// @param[in]     queryComponent  Query's current component id
/// @param[in]     nodeIdx         Index of the subtree root to visit (caller passes 0)
/// @param[in,out] bestMrd         Best MRD found so far (caller seeds with `+inf`)
/// @param[in,out] bestIdx         Index of the best different-component point so far
/// @param[in]     distFunc        Metric functor instance (unscaled metric)
/// @param[in]     invAlpha        `1.0 / alpha`, applied only to dist(q,p) inside MRD
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void nearestMrdBoruvkaQuery(const algorithmFPType * data, size_t nCols, const KdNode<algorithmFPType> * nodes, const DAAL_INT * pointIndices,
                                   const algorithmFPType * coreDistances, const algorithmFPType * bboxLo, const algorithmFPType * bboxHi,
                                   const algorithmFPType * minCoreDistNode, const DAAL_INT * componentOf, const algorithmFPType * queryPoint,
                                   DAAL_INT queryIdx, algorithmFPType queryCoreD, DAAL_INT queryComponent, DAAL_INT nodeIdx,
                                   algorithmFPType & bestMrd, DAAL_INT & bestIdx, const DistFunc & distFunc, algorithmFPType invAlpha)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    // Pruning 1: all points in this subtree belong to the same component as query
    if (node.componentId == queryComponent) return;

    // Pruning 2: bounding-box MRD lower bound
    {
        const algorithmFPType * lo    = bboxLo + nodeIdx * nCols;
        const algorithmFPType * hi    = bboxHi + nodeIdx * nCols;
        const algorithmFPType minDist = distFunc.template bboxLowerBound<cpu>(queryPoint, lo, hi, nCols);

        // MRD(q, p) = max(core_q, core_p, dist(q,p) * invAlpha)
        // >= max(core_q, minCoreDist_subtree, minDist * invAlpha)
        algorithmFPType mrdLB = minDist * invAlpha;
        if (queryCoreD > mrdLB) mrdLB = queryCoreD;
        if (minCoreDistNode[nodeIdx] > mrdLB) mrdLB = minCoreDistNode[nodeIdx];

        if (mrdLB >= bestMrd) return;
    }

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (DAAL_INT i = node.pointBegin; i < node.pointEnd; i++)
        {
            const DAAL_INT pi = pointIndices[i];
            if (componentOf[pi] == queryComponent) continue;

            const algorithmFPType * row = data + pi * nCols;
            const algorithmFPType dist  = distFunc.template pointDist<cpu>(queryPoint, row, nCols);

            algorithmFPType mrd = dist * invAlpha;
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

    // Visit nearer child first for tighter bestMrd
    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const DAAL_INT nearChild = (diff <= 0) ? node.left : node.right;
    const DAAL_INT farChild  = (diff <= 0) ? node.right : node.left;

    nearestMrdBoruvkaQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf,
                                                 queryPoint, queryIdx, queryCoreD, queryComponent, nearChild, bestMrd, bestIdx, distFunc, invAlpha);
    nearestMrdBoruvkaQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf,
                                                 queryPoint, queryIdx, queryCoreD, queryComponent, farChild, bestMrd, bestIdx, distFunc, invAlpha);
}

/// Compute core distances and the MST under MRD on a kd-tree, templated on metric.
///
/// Pipeline:
///   1. per-point k-NN query against the kd-tree -> `coreDistances`;
///   2. bottom-up reduction -> `minCoreDistNode`;
///   3. Boruvka rounds: per-point nearest-different-component MRD query,
///      reduce to per-component best edges, union via union-find, refresh
///      per-node component ids. Loops until a single component remains or no
///      progress is made.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
/// @tparam DistFunc        Metric functor
///
/// @param[in]     data            Row-major input buffer of size `nRows x nCols`
/// @param[in]     nRows           Number of points
/// @param[in]     nCols           Number of features
/// @param[in]     minSamples      Number of neighbors used for core distance (k)
/// @param[in,out] nodes           kd-tree nodes (component ids updated each round)
/// @param[in]     pointIndices    Point-index permutation produced by buildKdTree
/// @param[in]     totalTreeNodes  Number of kd-tree nodes
/// @param[in]     bboxLo          Per-node lower bbox bounds
/// @param[in]     bboxHi          Per-node upper bbox bounds
/// @param[out]    coreDistances   Per-point core distances, length `nRows`
/// @param[out]    mstFrom         Source endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstTo           Target endpoint per MST edge, length `nRows - 1`
/// @param[out]    mstWeights      Edge weights (MRD), length `nRows - 1`
/// @param[in]     distFunc        Metric functor instance (unscaled metric)
/// @param[in]     alpha           Robust single-linkage scaling factor; applied
///                                only to dist(q,p) inside MRD (not to k-NN
///                                core distances or to the metric used for tree
///                                queries)
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void computeCoreDistAndMst(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples, KdNode<algorithmFPType> * nodes,
                                  DAAL_INT * pointIndices, DAAL_INT totalTreeNodes, algorithmFPType * bboxLo, algorithmFPType * bboxHi,
                                  algorithmFPType * coreDistances, DAAL_INT * mstFrom, DAAL_INT * mstTo, algorithmFPType * mstWeights,
                                  const DistFunc & distFunc, double alpha)
{
    const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
    // Canonical HDBSCAN core distance (Campello 2013): the distance to the
    // `minSamples`-th nearest neighbor counting the query point itself as
    // neighbor #1. The kd-tree traversal pushes the query point into the heap
    // along with the other leaf points, so a heap of size `minSamples` holds
    // {self + (minSamples - 1) non-self}, and the heap top is the
    // `minSamples`-th-including-self answer.
    const DAAL_INT k = static_cast<DAAL_INT>(minSamples);

    // Step 2: Compute core distances via k-NN queries on the kd-tree
    daal::threader_for(nRows, nRows, [&](size_t i) {
        KnnHeap<algorithmFPType, cpu> heap(k);
        if (!heap.ok()) return;

        knnQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, data + i * nCols, 0, heap, distFunc);

        coreDistances[i] = heap.maxDist();
    });

    // Step 2b: Per-node minimum core distances
    TArrayScalable<algorithmFPType, cpu> minCoreDistNodeVec(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeVec.get();
    if (!minCoreDistNode) return;
    computeMinCoreDists<algorithmFPType, cpu>(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // Step 3: Boruvka MST

    TArray<DAAL_INT, cpu> ufParentVec(nRows);
    TArray<DAAL_INT, cpu> ufRankVec(nRows);
    TArray<DAAL_INT, cpu> componentOfVec(nRows);
    DAAL_INT * ufParent    = ufParentVec.get();
    DAAL_INT * ufRank      = ufRankVec.get();
    DAAL_INT * componentOf = componentOfVec.get();
    if (!ufParent || !ufRank || !componentOf) return;

    TArrayScalable<algorithmFPType, cpu> pointBestMrdVec(nRows);
    TArray<DAAL_INT, cpu> pointBestIdxVec(nRows);
    algorithmFPType * pointBestMrd = pointBestMrdVec.get();
    DAAL_INT * pointBestIdx        = pointBestIdxVec.get();
    if (!pointBestMrd || !pointBestIdx) return;

    TArrayScalable<algorithmFPType, cpu> compBestMrdVec(nRows);
    TArray<DAAL_INT, cpu> compBestFromVec(nRows);
    TArray<DAAL_INT, cpu> compBestToVec(nRows);
    algorithmFPType * compBestMrd = compBestMrdVec.get();
    DAAL_INT * compBestFrom       = compBestFromVec.get();
    DAAL_INT * compBestTo         = compBestToVec.get();
    if (!compBestMrd || !compBestFrom || !compBestTo) return;

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]    = static_cast<DAAL_INT>(i);
        componentOf[i] = static_cast<DAAL_INT>(i);
    }
    services::internal::service_memset_seq<DAAL_INT, cpu>(ufRank, 0, nRows);

    UnionFind uf { ufParent, ufRank };

    updateNodeComponents<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);

    size_t edgesAdded    = 0;
    size_t numComponents = nRows;

    // Only phase 1 (nearest-different-component MRD tree query) is
    // method-specific; phases 2-4 route through hdbscan_boruvka_utils.h.
    while (numComponents > 1)
    {
        daal::threader_for(nRows, nRows, [&](size_t i) {
            const DAAL_INT comp              = componentOf[i];
            const algorithmFPType * queryPt  = data + i * nCols;
            const algorithmFPType queryCoreD = coreDistances[i];
            algorithmFPType bestMrd          = daal::services::internal::MaxVal<algorithmFPType>::get();
            DAAL_INT bestIdx                 = -1;

            nearestMrdBoruvkaQuery<algorithmFPType, cpu>(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode,
                                                         componentOf, queryPt, static_cast<DAAL_INT>(i), queryCoreD, comp, 0, bestMrd, bestIdx,
                                                         distFunc, invAlpha);

            pointBestMrd[i] = bestMrd;
            pointBestIdx[i] = bestIdx;
        });

        reduceComponentBestEdges<algorithmFPType>(nRows, componentOf, pointBestMrd, pointBestIdx, compBestMrd, compBestFrom, compBestTo);

        const size_t addedThisRound = mergeComponentsEmitEdges<algorithmFPType>(nRows, compBestMrd, compBestFrom, compBestTo, uf, mstFrom, mstTo,
                                                                                mstWeights, edgesAdded, numComponents);

        if (addedThisRound == 0) break;

        refreshComponentIds<cpu>(nRows, uf, componentOf);

        updateNodeComponents<algorithmFPType, cpu>(nodes, pointIndices, componentOf, 0);
    }
}

// =========================================================================
// Main HDBSCAN kd-tree compute
// =========================================================================
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

    // Label output is stored as `int32_t` in the assignments NumericTable
    // (codebase-wide DAAL convention shared with kmeans / knn /
    // decision_forest / etc.). Refuse inputs where the label count could
    // exceed the destination-type bound. The label count is bounded above by
    // the number of surviving clusters, itself bounded by `nRows / mcs`.
    // Guard against `INT32_MAX` (not `INT_MAX`) because the storage type is
    // fixed-width `int32_t` regardless of the data model -- on a hypothetical
    // ILP64 platform `INT_MAX` would be 2^63 - 1 and would let overflowing
    // inputs through.
    if (nRows / minClusterSize > static_cast<size_t>(INT32_MAX))
    {
        return services::Status(services::ErrorIncorrectSizeOfInputNumericTable);
    }

    const size_t edgeCount = nRows - 1;

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // =========================================================================
    // Step 1: Build k-d tree with bounding boxes
    // =========================================================================

    const DAAL_INT maxLeafSize = static_cast<DAAL_INT>(leafSize);
    // Conservative upper bound: binary tree stopping at maxLeafSize > 1 has
    // <= 2*nRows - 1 nodes; 4*nRows removes sensitivity to pathological splits.
    const DAAL_INT maxNodes = 4 * static_cast<DAAL_INT>(nRows);

    TArray<KdNode<algorithmFPType>, cpu> nodesVec(maxNodes);
    KdNode<algorithmFPType> * nodes = nodesVec.get();
    DAAL_CHECK_MALLOC(nodes);

    TArray<DAAL_INT, cpu> pointIndicesVec(nRows);
    DAAL_INT * pointIndices = pointIndicesVec.get();
    DAAL_CHECK_MALLOC(pointIndices);
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<DAAL_INT>(i);

    // Bounding boxes stored in SoA layout: bboxLo[nodeIdx * nCols + d]
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, maxNodes, nCols);
    TArray<algorithmFPType, cpu> bboxLoVec(static_cast<size_t>(maxNodes) * nCols);
    TArray<algorithmFPType, cpu> bboxHiVec(static_cast<size_t>(maxNodes) * nCols);
    algorithmFPType * bboxLo = bboxLoVec.get();
    algorithmFPType * bboxHi = bboxHiVec.get();
    DAAL_CHECK_MALLOC(bboxLo);
    DAAL_CHECK_MALLOC(bboxHi);

    DAAL_INT nextNode = 0;
    buildKdTree<algorithmFPType, cpu>(data, pointIndices, 0, static_cast<DAAL_INT>(nRows), nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);
    const DAAL_INT totalTreeNodes = nextNode;

    // =========================================================================
    // Steps 2-3: Core distances + Boruvka MST (dispatched by metric)
    // =========================================================================

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

    using algorithms::internal::PairwiseDistanceType;

    // Robust single linkage: alpha is applied only to dist(q,p) inside MRD
    // (canonical HDBSCAN). The metric used for k-NN core distances and for
    // kd-tree pruning is left unscaled.
    //
    // The upstream oneAPI check_preconditions() in detail/compute_ops.hpp
    // rejects cosine with method::kd_tree (kd-tree pruning requires an L_p
    // distance). The explicit `case cosine:` below is a defense-in-depth
    // guard so if the DAAL kernel is ever reached with cosine + kd_tree
    // (e.g. via a future direct-DAAL entry point that bypasses the oneAPI
    // check), we fail loudly with ErrorMethodNotSupported rather than
    // silently routing to euclidean via a `default:` fall-through.
    switch (pairwiseDistance)
    {
    case PairwiseDistanceType::euclidean:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, EuclideanDist<algorithmFPType> {}, alpha);
        break;
    case PairwiseDistanceType::manhattan:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, ManhattanDist<algorithmFPType> {}, alpha);
        break;
    case PairwiseDistanceType::minkowski:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, MinkowskiDist<algorithmFPType>(minkowskiDegree),
                                                    alpha);
        break;
    case PairwiseDistanceType::chebyshev:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, ChebyshevDist<algorithmFPType> {}, alpha);
        break;
    case PairwiseDistanceType::cosine:
    default: return services::Status(services::ErrorMethodNotSupported);
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters (shared with brute_force)
    // =========================================================================

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
