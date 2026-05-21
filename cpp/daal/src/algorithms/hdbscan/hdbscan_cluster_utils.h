/* file: hdbscan_cluster_utils.h */
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

#ifndef __HDBSCAN_CLUSTER_UTILS_H__
#define __HDBSCAN_CLUSTER_UTILS_H__

#include "src/algorithms/service_sort.h"
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
using daal::services::internal::MaxVal;
using daal::services::internal::TArray;

/// One edge of the condensed cluster tree.
///
/// Produced by buildCondensedTree: an edge connects a parent cluster id to either
/// a child cluster id (with childSize == subtree size) or a single fallen-out
/// point id (childSize == 1).
struct CondensedEdge
{
    int parent;    ///< Parent cluster id
    int child;     ///< Child cluster id or fallen-out point id (< nRows)
    int childSize; ///< Number of original points in the child subtree (1 for fallen leaves)
};

/// Sort MST edges in ascending order of weight, keeping endpoint arrays aligned.
///
/// Uses the shared three-array qSort: mstWeights is the key, mstFrom/mstTo are
/// permuted in lock-step. Required by buildDendrogramFromSortedMst, whose
/// union-find merge order assumes ascending weights.
///
/// @tparam algorithmFPType Floating-point type used for edge weights
/// @tparam cpu             CPU dispatch tag
///
/// @param[in,out] mstFrom    Source endpoint of each MST edge, length `edgeCount`
/// @param[in,out] mstTo      Target endpoint of each MST edge, length `edgeCount`
/// @param[in,out] mstWeights Edge weights (sort key), length `edgeCount`
/// @param[in]     edgeCount  Number of MST edges
template <typename algorithmFPType, CpuType cpu>
static void sortMstEdges(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t edgeCount)
{
    daal::algorithms::internal::qSort<algorithmFPType, int, int, cpu>(edgeCount, mstWeights, mstFrom, mstTo);
}

/// Build the single-linkage dendrogram from sorted MST edges via union-find.
///
/// Each MST edge in ascending-weight order merges two components into a new
/// internal node; the resulting tree has `edgeCount` internal nodes indexed
/// `[nRows, nRows + edgeCount)` and the `nRows` original points as leaves
/// indexed `[0, nRows)`.
///
/// @tparam algorithmFPType Floating-point type used for edge weights
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  mstFrom    Source endpoint of each MST edge, length `edgeCount`
/// @param[in]  mstTo      Target endpoint of each MST edge, length `edgeCount`
/// @param[in]  mstWeights Edge weights, sorted ascending, length `edgeCount`
/// @param[in]  nRows      Number of original points (leaf node count)
/// @param[in]  edgeCount  Number of MST edges (`nRows - 1` for a connected MST)
/// @param[out] nodeSize   Subtree size for every node, length `totalNodes`
/// @param[out] leftChild  Left child id for every internal node, length `totalNodes`
/// @param[out] rightChild Right child id for every internal node, length `totalNodes`
/// @param[out] nodeWeight Edge weight that created each internal node, length `totalNodes`
/// @param[in]  totalNodes Size of every output array (`2 * nRows - 1`)
///
/// @return Root node id, or -1 if the MST is empty
template <typename algorithmFPType, CpuType cpu>
static int buildDendrogramFromSortedMst(const int * mstFrom, const int * mstTo, const algorithmFPType * mstWeights, size_t nRows, size_t edgeCount,
                                        int * nodeSize, int * leftChild, int * rightChild, algorithmFPType * nodeWeight, size_t totalNodes)
{
    TArray<int, cpu> ufParentArr(nRows);
    TArray<int, cpu> compSizeArr(nRows);
    TArray<int, cpu> compToNodeArr(nRows);
    int * ufParent   = ufParentArr.get();
    int * compSize   = compSizeArr.get();
    int * compToNode = compToNodeArr.get();

    for (size_t i = 0; i < totalNodes; i++)
    {
        nodeSize[i]   = (i < nRows) ? 1 : 0;
        leftChild[i]  = -1;
        rightChild[i] = -1;
        nodeWeight[i] = algorithmFPType(0);
    }
    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]   = static_cast<int>(i);
        compSize[i]   = 1;
        compToNode[i] = static_cast<int>(i);
    }

    auto ufFind = [&](int x) -> int {
        while (ufParent[x] != x)
        {
            ufParent[x] = ufParent[ufParent[x]];
            x           = ufParent[x];
        }
        return x;
    };

    int root = -1;
    for (size_t e = 0; e < edgeCount; e++)
    {
        const int ru = ufFind(mstFrom[e]);
        const int rv = ufFind(mstTo[e]);
        if (ru == rv) continue;

        const int nodeId  = static_cast<int>(nRows + e);
        const int newSize = compSize[ru] + compSize[rv];

        leftChild[nodeId]  = compToNode[ru];
        rightChild[nodeId] = compToNode[rv];
        nodeWeight[nodeId] = mstWeights[e];
        nodeSize[nodeId]   = newSize;
        root               = nodeId;

        if (compSize[ru] < compSize[rv])
        {
            ufParent[ru]   = rv;
            compSize[rv]   = newSize;
            compToNode[rv] = nodeId;
        }
        else
        {
            ufParent[rv]   = ru;
            compSize[ru]   = newSize;
            compToNode[ru] = nodeId;
        }
    }

    return root;
}

/// Build the condensed cluster tree from a single-linkage dendrogram.
///
/// Walks the dendrogram top-down. At each internal node, sides whose subtree
/// size is at least `mcs` (min cluster size) keep their cluster id; sides
/// smaller than `mcs` are emitted as "fallen" point edges with the parent
/// cluster id and the death lambda `1 / nodeWeight[nid]`. New cluster ids are
/// allocated from `nextCid` only when both sides survive (a real split).
///
/// @tparam algorithmFPType Floating-point type used for edge weights
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     root            Dendrogram root node id (output of buildDendrogramFromSortedMst)
/// @param[in]     nRows           Number of original points (leaf count)
/// @param[in]     mcs             Minimum cluster size threshold
/// @param[in]     nodeSize        Subtree size for every node, length `2*nRows - 1`
/// @param[in]     leftChild       Left child id for every internal node, length `2*nRows - 1`
/// @param[in]     rightChild      Right child id for every internal node, length `2*nRows - 1`
/// @param[in]     nodeWeight      Edge weight that created each internal node, length `2*nRows - 1`
/// @param[in,out] dendroToCluster Maps dendrogram node id -> cluster id (root preset; updated in place)
/// @param[out]    condensed       Output condensed-tree edges, capacity at least `3*nRows`
/// @param[out]    condensedLambda Death lambda for each emitted edge, capacity at least `3*nRows`
/// @param[in,out] nextCid         Cluster-id allocator (caller seeds with `nRows`; advanced on each split)
///
/// @return Number of edges written to `condensed` / `condensedLambda`
template <typename algorithmFPType, CpuType cpu>
static size_t buildCondensedTree(int root, size_t nRows, int mcs, const int * nodeSize, const int * leftChild, const int * rightChild,
                                 const algorithmFPType * nodeWeight, int * dendroToCluster, CondensedEdge * condensed,
                                 algorithmFPType * condensedLambda, int & nextCid)
{
    TArray<int, cpu> leafStackArr(nRows);
    TArray<int, cpu> fallenBufArr(nRows);
    int * leafStack = leafStackArr.get();
    int * fallenBuf = fallenBufArr.get();

    auto collectLeaves = [&](int startNid, int * out, size_t & outCount) {
        outCount              = 0;
        int stackTop          = 0;
        leafStack[stackTop++] = startNid;
        while (stackTop > 0)
        {
            const int nid = leafStack[--stackTop];
            if (nid < static_cast<int>(nRows))
            {
                out[outCount++] = nid;
            }
            else
            {
                if (rightChild[nid] >= 0) leafStack[stackTop++] = rightChild[nid];
                if (leftChild[nid] >= 0) leafStack[stackTop++] = leftChild[nid];
            }
        }
    };

    struct StackItem
    {
        int node;
        int cluster;
    };
    const size_t nDendroNodes = nRows - 1;
    TArray<StackItem, cpu> mainStackArr(nDendroNodes + 1);
    StackItem * mainStack     = mainStackArr.get();
    int mainStackTop          = 0;
    mainStack[mainStackTop++] = { root, dendroToCluster[root] };

    size_t nCondensed = 0;

    auto emitFallenLeaves = [&](int subtree, int parentCid, algorithmFPType lambda) {
        size_t nFallen = 0;
        collectLeaves(subtree, fallenBuf, nFallen);
        for (size_t fi = 0; fi < nFallen; fi++)
        {
            condensed[nCondensed]       = { parentCid, fallenBuf[fi], 1 };
            condensedLambda[nCondensed] = lambda;
            nCondensed++;
        }
    };

    while (mainStackTop > 0)
    {
        const StackItem item = mainStack[--mainStackTop];
        const int nid        = item.node;
        const int parentCid  = item.cluster;

        if (nid < static_cast<int>(nRows)) continue;

        const int lc = leftChild[nid];
        const int rc = rightChild[nid];
        if (lc < 0 || rc < 0) continue;

        const int ls                 = nodeSize[lc];
        const int rs                 = nodeSize[rc];
        const algorithmFPType lambda = (nodeWeight[nid] > algorithmFPType(0)) ? algorithmFPType(1) / nodeWeight[nid] : MaxVal<algorithmFPType>::get();

        const bool lBig = ls >= mcs;
        const bool rBig = rs >= mcs;

        if (lBig && rBig)
        {
            const int lcid              = nextCid++;
            const int rcid              = nextCid++;
            dendroToCluster[lc]         = lcid;
            dendroToCluster[rc]         = rcid;
            condensed[nCondensed]       = { parentCid, lcid, ls };
            condensedLambda[nCondensed] = lambda;
            nCondensed++;
            condensed[nCondensed]       = { parentCid, rcid, rs };
            condensedLambda[nCondensed] = lambda;
            nCondensed++;
            mainStack[mainStackTop++] = { lc, lcid };
            mainStack[mainStackTop++] = { rc, rcid };
        }
        else if (lBig)
        {
            dendroToCluster[lc] = parentCid;
            emitFallenLeaves(rc, parentCid, lambda);
            mainStack[mainStackTop++] = { lc, parentCid };
        }
        else if (rBig)
        {
            dendroToCluster[rc] = parentCid;
            emitFallenLeaves(lc, parentCid, lambda);
            mainStack[mainStackTop++] = { rc, parentCid };
        }
        else
        {
            emitFallenLeaves(lc, parentCid, lambda);
            emitFallenLeaves(rc, parentCid, lambda);
        }
    }

    return nCondensed;
}

/// Initialize per-cluster bookkeeping arrays from the condensed tree.
///
/// One pass over the condensed edges; for every cluster->cluster edge it sets
/// the child's birth lambda, marks the parent as non-leaf, increments the
/// parent's child count, and records the child's subtree size. Per-cluster
/// arrays are zeroed first; the root cluster's size is preset to `nRows`.
///
/// @tparam algorithmFPType Floating-point type used for cluster lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed       Condensed-tree edges, length `nCondensed`
/// @param[in]  condensedLambda Death lambda per edge, length `nCondensed`
/// @param[in]  nCondensed      Number of condensed-tree edges
/// @param[in]  nRows           Number of original points
/// @param[in]  nClusters       Total cluster count (next free cluster id)
/// @param[in]  rootCid         Root cluster id (== `nRows`)
/// @param[out] lambdaBirth     Birth lambda per cluster, length `nClusters`
/// @param[out] isLeafCluster   1 if cluster has no cluster-children, else 0; length `nClusters`
/// @param[out] clusterSz       Number of points in each cluster, length `nClusters`
/// @param[out] childCount      Number of cluster-children per cluster, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void initClusterMetadata(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                                int nClusters, int rootCid, algorithmFPType * lambdaBirth, char * isLeafCluster, int * clusterSz, int * childCount)
{
    for (int c = 0; c < nClusters; c++)
    {
        lambdaBirth[c]   = algorithmFPType(0);
        isLeafCluster[c] = 1;
        clusterSz[c]     = 0;
        childCount[c]    = 0;
    }
    clusterSz[rootCid] = static_cast<int>(nRows);

    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows))
        {
            lambdaBirth[e.child]    = condensedLambda[ei];
            isLeafCluster[e.parent] = 0;
            childCount[e.parent]++;
            clusterSz[e.child] = e.childSize;
        }
    }
}

/// Accumulate HDBSCAN stability scores for every cluster.
///
/// stability[c] = sum over edges (parent==c) of (deathLambda - birthLambda) * childSize.
/// Negative contributions (death before birth, possible at the root) are clamped to 0.
/// Used by Excess of Mass cluster selection.
///
/// @tparam algorithmFPType Floating-point type used for cluster lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed       Condensed-tree edges, length `nCondensed`
/// @param[in]  condensedLambda Death lambda per edge, length `nCondensed`
/// @param[in]  nCondensed      Number of condensed-tree edges
/// @param[in]  nClusters       Total cluster count
/// @param[in]  lambdaBirth     Birth lambda per cluster, length `nClusters`
/// @param[out] stability       Accumulated stability per cluster, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void computeClusterStability(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, int nClusters,
                                    const algorithmFPType * lambdaBirth, algorithmFPType * stability)
{
    for (int c = 0; c < nClusters; c++) stability[c] = algorithmFPType(0);
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e       = condensed[ei];
        const algorithmFPType birth   = lambdaBirth[e.parent];
        const algorithmFPType contrib = (condensedLambda[ei] - birth) * static_cast<algorithmFPType>(e.childSize);
        if (contrib > algorithmFPType(0)) stability[e.parent] += contrib;
    }
}

/// Run the Excess-of-Mass cluster selection pass on the condensed tree.
///
/// Iterates clusters in decreasing id order (children before parents). For each
/// non-leaf cluster, compares the parent's stability against the sum of its
/// children's stabilities. If children win, the parent is unselected and its
/// stability is replaced by the sum (so its grandparent sees the propagated
/// score). If the parent wins, every descendant is unselected via an explicit
/// stack walk over `childOffset`/`childList`. Clusters whose size exceeds
/// `mcsMax` get parentStab forced to 0, so they cannot beat their children.
///
/// @tparam algorithmFPType Floating-point type used for cluster stabilities
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     nClusters     Total cluster count
/// @param[in]     rootCid       Root cluster id (== `nRows`)
/// @param[in]     mcsMax        Maximum allowed cluster size (`MaxVal<int>` if uncapped)
/// @param[in,out] stability     Per-cluster stability; updated in place with propagated child sums
/// @param[in]     clusterSz     Per-cluster point counts, length `nClusters`
/// @param[in]     isLeafCluster Leaf-cluster mask, length `nClusters`
/// @param[in]     childOffset   CSR offsets into `childList`, length `nClusters + 1`
/// @param[in]     childCount    Per-cluster cluster-child counts, length `nClusters`
/// @param[in]     childList     CSR child cluster ids, length `childOffset[nClusters]`
/// @param[in,out] descStack     Scratch stack for the descendant-unselect walk, length `nClusters`
/// @param[in,out] isSelected    Selection mask updated in place, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void runEomSelection(int nClusters, int rootCid, int mcsMax, algorithmFPType * stability, const int * clusterSz, const char * isLeafCluster,
                            const int * childOffset, const int * childCount, const int * childList, int * descStack, char * isSelected)
{
    for (int c = nClusters - 1; c >= rootCid; c--)
    {
        if (isLeafCluster[c]) continue;

        algorithmFPType childSum = algorithmFPType(0);
        for (int ci = childOffset[c]; ci < childOffset[c] + childCount[c]; ci++) childSum += stability[childList[ci]];

        const algorithmFPType parentStab = (clusterSz[c] > mcsMax) ? algorithmFPType(0) : stability[c];

        if (childSum > parentStab)
        {
            isSelected[c] = 0;
            stability[c]  = childSum;
        }
        else
        {
            int descTop = 0;
            for (int ci = childOffset[c]; ci < childOffset[c] + childCount[c]; ci++) descStack[descTop++] = childList[ci];
            while (descTop > 0)
            {
                const int d   = descStack[--descTop];
                isSelected[d] = 0;
                for (int ci = childOffset[d]; ci < childOffset[d] + childCount[d]; ci++) descStack[descTop++] = childList[ci];
            }
        }
    }
}

/// Promote selected clusters that are too dense (birth distance < epsilon) to their parent.
///
/// Implements the cluster_selection_epsilon refinement: any selected cluster
/// whose birth distance `1 / lambdaBirth[c]` is below `clusterSelectionEpsilon`
/// is unselected and its parent is selected instead. Repeats until no further
/// changes (the parent itself may then be too dense and get promoted again).
///
/// @tparam algorithmFPType Floating-point type used for cluster lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     condensed              Condensed-tree edges, length `nCondensed`
/// @param[in]     nCondensed             Number of condensed-tree edges
/// @param[in]     nRows                  Number of original points (used to test child < nRows)
/// @param[in]     nClusters              Total cluster count
/// @param[in]     rootCid                Root cluster id (== `nRows`)
/// @param[in]     lambdaBirth            Birth lambda per cluster, length `nClusters`
/// @param[in]     clusterSelectionEpsilon Distance threshold; clusters with birth distance below this are merged into parent
/// @param[in,out] isSelected             Selection mask updated in place, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void applyClusterSelectionEpsilon(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, int nClusters, int rootCid,
                                         const algorithmFPType * lambdaBirth, double clusterSelectionEpsilon, char * isSelected)
{
    TArray<int, cpu> clusterParentArr(nClusters);
    int * clusterParent = clusterParentArr.get();
    for (int c = 0; c < nClusters; c++) clusterParent[c] = -1;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows)) clusterParent[e.child] = e.parent;
    }

    const algorithmFPType eps = static_cast<algorithmFPType>(clusterSelectionEpsilon);
    bool changed              = true;
    while (changed)
    {
        changed = false;
        for (int c = rootCid + 1; c < nClusters; c++)
        {
            if (!isSelected[c]) continue;
            const algorithmFPType birthDist = (lambdaBirth[c] > algorithmFPType(0)) ? algorithmFPType(1) / lambdaBirth[c] : algorithmFPType(0);
            if (birthDist < eps)
            {
                const int parent = clusterParent[c];
                if (parent >= rootCid && parent < nClusters)
                {
                    isSelected[c]      = 0;
                    isSelected[parent] = 1;
                    changed            = true;
                }
            }
        }
    }
}

/// Build CSR-style child offsets via prefix-sum over per-cluster child counts.
///
/// @param[in]  nClusters   Total cluster count
/// @param[in]  childCount  Per-cluster cluster-child counts, length `nClusters`
/// @param[out] childOffset Prefix sums; `childOffset[c]` is the start of c's children, length `nClusters + 1`
static void computeChildOffsets(int nClusters, const int * childCount, int * childOffset)
{
    childOffset[0] = 0;
    for (int c = 1; c <= nClusters; c++) childOffset[c] = childOffset[c - 1] + childCount[c - 1];
}

/// Fill the CSR child list for every cluster from the condensed tree.
///
/// Pairs with computeChildOffsets: emits one entry per cluster->cluster edge,
/// using a per-cluster cursor so writes for the same parent are appended in
/// the order they appear in `condensed`.
///
/// @tparam algorithmFPType Floating-point type (unused; kept for cpu dispatch)
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed   Condensed-tree edges, length `nCondensed`
/// @param[in]  nCondensed  Number of condensed-tree edges
/// @param[in]  nRows       Number of original points (used to test child < nRows)
/// @param[in]  nClusters   Total cluster count
/// @param[in]  childOffset CSR offsets, length `nClusters + 1`
/// @param[out] childList   Flat list of child cluster ids, length `childOffset[nClusters]`
template <typename algorithmFPType, CpuType cpu>
static void fillChildList(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, int nClusters, const int * childOffset, int * childList)
{
    TArray<int, cpu> fillCursorArr(nClusters);
    int * fillCursor = fillCursorArr.get();
    for (int c = 0; c < nClusters; c++) fillCursor[c] = 0;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows))
        {
            childList[childOffset[e.parent] + fillCursor[e.parent]] = e.child;
            fillCursor[e.parent]++;
        }
    }
}

/// Reject the "single-cluster" outcome when `allow_single_cluster=false`.
///
/// If EOM selected only the root cluster and the root has cluster-children,
/// unselect the root and select all its direct children instead. No-op
/// otherwise.
///
/// @param[in]     nClusters     Total cluster count
/// @param[in]     rootCid       Root cluster id
/// @param[in]     isLeafCluster Leaf-cluster mask, length `nClusters`
/// @param[in]     childOffset   CSR child offsets, length `nClusters + 1`
/// @param[in]     childCount    Per-cluster cluster-child counts, length `nClusters`
/// @param[in]     childList     CSR child cluster ids, length `childOffset[nClusters]`
/// @param[in,out] isSelected    Selection mask updated in place, length `nClusters`
static void enforceAllowSingleCluster(int nClusters, int rootCid, const char * isLeafCluster, const int * childOffset, const int * childCount,
                                      const int * childList, char * isSelected)
{
    int selectedCount = 0;
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) selectedCount++;
    }
    if (selectedCount == 1 && isSelected[rootCid] && !isLeafCluster[rootCid])
    {
        isSelected[rootCid] = 0;
        for (int ci = childOffset[rootCid]; ci < childOffset[rootCid] + childCount[rootCid]; ci++)
        {
            isSelected[childList[ci]] = 1;
        }
    }
}

/// Top-level cluster selection on the condensed tree.
///
/// Allocates the per-cluster bookkeeping arrays (lambdaBirth, isLeafCluster,
/// clusterSz, childCount, childOffset/childList, stability), seeds isSelected
/// with the size-feasible mask, then runs:
///   - leaf-mode (clusterSelection == 1): pick every leaf cluster of size >= mcs;
///   - EOM-mode (default): runEomSelection;
/// followed by enforceAllowSingleCluster (if `allowSingleCluster=false`) and
/// applyClusterSelectionEpsilon (if `clusterSelectionEpsilon > 0`).
///
/// @tparam algorithmFPType Floating-point type used for cluster lambdas/stabilities
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed              Condensed-tree edges, length `nCondensed`
/// @param[in]  condensedLambda        Death lambda per edge, length `nCondensed`
/// @param[in]  nCondensed             Number of condensed-tree edges
/// @param[in]  nRows                  Number of original points
/// @param[in]  nClusters              Total cluster count (next free cluster id)
/// @param[in]  rootCid                Root cluster id (== `nRows`)
/// @param[in]  mcs                    Minimum cluster size threshold
/// @param[in]  maxClusterSize         Maximum cluster size cap (0 == uncapped)
/// @param[in]  clusterSelection       0 = EOM, 1 = leaf
/// @param[in]  allowSingleCluster     If false, reject root-only outcomes
/// @param[in]  clusterSelectionEpsilon Distance epsilon for cluster_selection_epsilon refinement (0 == disabled)
/// @param[out] isSelected             Final selection mask, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void selectClusters(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows, int nClusters,
                           int rootCid, int mcs, size_t maxClusterSize, int clusterSelection, bool allowSingleCluster, double clusterSelectionEpsilon,
                           char * isSelected)
{
    TArray<algorithmFPType, cpu> stabilityArr(nClusters);
    TArray<algorithmFPType, cpu> lambdaBirthArr(nClusters);
    TArray<char, cpu> isLeafClusterArr(nClusters);
    TArray<int, cpu> clusterSzArr(nClusters);
    TArray<int, cpu> childCountArr(nClusters);
    algorithmFPType * stability   = stabilityArr.get();
    algorithmFPType * lambdaBirth = lambdaBirthArr.get();
    char * isLeafCluster          = isLeafClusterArr.get();
    int * clusterSz               = clusterSzArr.get();
    int * childCount              = childCountArr.get();

    initClusterMetadata<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, lambdaBirth, isLeafCluster,
                                              clusterSz, childCount);

    TArray<int, cpu> childOffsetArr(nClusters + 1);
    int * childOffset = childOffsetArr.get();
    computeChildOffsets(nClusters, childCount, childOffset);
    const int totalChildren = childOffset[nClusters];

    TArray<int, cpu> childListArr(totalChildren > 0 ? totalChildren : 1);
    int * childList = childListArr.get();
    fillChildList<algorithmFPType, cpu>(condensed, nCondensed, nRows, nClusters, childOffset, childList);

    computeClusterStability<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nClusters, lambdaBirth, stability);

    const int mcsMax = (maxClusterSize > 0) ? static_cast<int>(maxClusterSize) : MaxVal<int>::get();

    for (int c = 0; c < nClusters; c++)
    {
        isSelected[c] = (c >= rootCid && clusterSz[c] >= mcs) ? 1 : 0;
    }

    if (clusterSelection == 1)
    {
        for (int c = rootCid; c < nClusters; c++)
        {
            isSelected[c] = (isLeafCluster[c] && clusterSz[c] >= mcs) ? 1 : 0;
        }
    }
    else
    {
        TArray<int, cpu> descStackArr(nClusters);
        int * descStack = descStackArr.get();
        runEomSelection<algorithmFPType, cpu>(nClusters, rootCid, mcsMax, stability, clusterSz, isLeafCluster, childOffset, childCount, childList,
                                              descStack, isSelected);
    }

    if (!allowSingleCluster)
    {
        enforceAllowSingleCluster(nClusters, rootCid, isLeafCluster, childOffset, childCount, childList, isSelected);
    }

    if (clusterSelectionEpsilon > 0.0)
    {
        applyClusterSelectionEpsilon<algorithmFPType, cpu>(condensed, nCondensed, nRows, nClusters, rootCid, lambdaBirth, clusterSelectionEpsilon,
                                                           isSelected);
    }
}

/// Resolve a final point label for every cluster in one O(nClusters) forward sweep.
///
/// Invariant: in the condensed tree built by buildCondensedTree, a parent's
/// cluster id is always strictly less than each child's id (parents are emitted
/// before children). Iterating c = rootCid..nClusters-1 therefore visits
/// parents before children, and resolvedLabel[clusterParent[c]] is already
/// final by the time we read it. Replaces a per-point ancestor walk with a
/// single table lookup in labelPoints.
///
/// @param[in]  rootCid       Root cluster id (== `nRows`)
/// @param[in]  nClusters     Total cluster count
/// @param[in]  isSelected    Selection mask, length `nClusters`
/// @param[in]  clusterLabel  Dense label assigned to each selected cluster (-1 if unselected), length `nClusters`
/// @param[in]  clusterParent Parent cluster id per cluster (-1 for root), length `nClusters`
/// @param[out] resolvedLabel Final label for every cluster (-1 if no selected ancestor), length `nClusters`
static void resolveClusterLabels(int rootCid, int nClusters, const char * isSelected, const int * clusterLabel, const int * clusterParent,
                                 int * resolvedLabel)
{
    for (int c = 0; c < rootCid; c++) resolvedLabel[c] = -1;
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c])
        {
            resolvedLabel[c] = clusterLabel[c];
            continue;
        }
        const int p      = clusterParent[c];
        resolvedLabel[c] = (p >= rootCid && p < nClusters) ? resolvedLabel[p] : -1;
    }
}

/// Reverse the child arrays into a parent pointer for every dendrogram node.
///
/// @param[in]  leftChild    Left child id for every internal node, length `totalNodes`
/// @param[in]  rightChild   Right child id for every internal node, length `totalNodes`
/// @param[in]  nRows        Number of leaves (internal nodes start at index `nRows`)
/// @param[in]  totalNodes   Total node count (`2 * nRows - 1`)
/// @param[out] dendroParent Parent id per node (-1 for root and unused slots), length `totalNodes`
static void buildDendroParent(const int * leftChild, const int * rightChild, size_t nRows, size_t totalNodes, int * dendroParent)
{
    for (size_t i = 0; i < totalNodes; i++) dendroParent[i] = -1;
    for (size_t nid = nRows; nid < totalNodes; nid++)
    {
        if (leftChild[nid] >= 0) dendroParent[leftChild[nid]] = static_cast<int>(nid);
        if (rightChild[nid] >= 0) dendroParent[rightChild[nid]] = static_cast<int>(nid);
    }
}

/// Final point labeling phase. For each input point i, assign assignments[i] to
///   - the dense label (0..nLabels-1) of the deepest selected ancestor cluster
///     in the condensed tree, OR
///   - -1 if no such ancestor exists (the point is "noise").
///
/// Two passes over the points, both parallelized via daal::threader_for:
///   1) Points that fell out of a cluster directly (pointFellFrom[i] >= 0):
///      look up the resolved label of their drop cluster in O(1).
///   2) Points that never fell out before the root cluster: walk up the
///      dendrogram via dendroParent[] until hitting a node with a known
///      cluster id, then look up its resolved label.
/// resolveClusterLabels precomputes the dense label per cluster in one
/// O(nClusters) sweep so neither per-point pass walks the cluster tree.
///
/// @tparam algorithmFPType Floating-point type (unused; kept for cpu dispatch)
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed       Condensed-tree edges, length `nCondensed`
/// @param[in]  nCondensed      Number of condensed-tree edges
/// @param[in]  nRows           Number of original points
/// @param[in]  leftChild       Left child id per dendrogram node, length `2*nRows - 1`
/// @param[in]  rightChild      Right child id per dendrogram node, length `2*nRows - 1`
/// @param[in]  dendroToCluster Maps dendrogram node id -> cluster id (-1 if no cluster), length `2*nRows - 1`
/// @param[in]  isSelected      Selection mask, length `nClusters`
/// @param[in]  nClusters       Total cluster count
/// @param[in]  rootCid         Root cluster id (== `nRows`)
/// @param[out] assignments     Output point->label table, length `nRows`
///
/// @return Number of distinct labels emitted (0..labelCounter-1)
template <typename algorithmFPType, CpuType cpu>
static int labelPoints(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, const int * leftChild, const int * rightChild,
                       const int * dendroToCluster, const char * isSelected, int nClusters, int rootCid, int * assignments)
{
    const size_t totalNodes = 2 * nRows - 1;

    int labelCounter = 0;
    TArray<int, cpu> clusterLabelArr(nClusters);
    int * clusterLabel = clusterLabelArr.get();
    for (int c = 0; c < nClusters; c++) clusterLabel[c] = -1;
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }

    TArray<int, cpu> clusterParentArr(nClusters);
    TArray<int, cpu> pointFellFromArr(nRows);
    int * clusterParent = clusterParentArr.get();
    int * pointFellFrom = pointFellFromArr.get();
    for (int c = 0; c < nClusters; c++) clusterParent[c] = -1;
    for (size_t i = 0; i < nRows; i++) pointFellFrom[i] = -1;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows))
            clusterParent[e.child] = e.parent;
        else
            pointFellFrom[e.child] = e.parent;
    }

    TArray<int, cpu> resolvedLabelArr(nClusters);
    int * resolvedLabel = resolvedLabelArr.get();
    resolveClusterLabels(rootCid, nClusters, isSelected, clusterLabel, clusterParent, resolvedLabel);

    const int iNRows = static_cast<int>(nRows);
    daal::threader_for(iNRows, iNRows, [&](size_t i) {
        const int c    = pointFellFrom[i];
        assignments[i] = (c >= rootCid && c < nClusters) ? resolvedLabel[c] : -1;
    });

    TArray<int, cpu> dendroParentArr(totalNodes);
    int * dendroParent = dendroParentArr.get();
    buildDendroParent(leftChild, rightChild, nRows, totalNodes, dendroParent);

    daal::threader_for(iNRows, iNRows, [&](size_t i) {
        if (pointFellFrom[i] >= 0) return;

        int nid = static_cast<int>(i);
        while (nid >= 0 && nid < static_cast<int>(totalNodes))
        {
            const int cid = dendroToCluster[nid];
            if (cid >= rootCid)
            {
                assignments[i] = resolvedLabel[cid];
                break;
            }
            nid = dendroParent[nid];
        }
    });

    return labelCounter;
}

/// Sort MST edges by weight and extract flat clusters via condensed tree + EOM.
///
/// Shared end-of-pipeline used by brute_force, kd_tree, and ball_tree methods.
/// Sequence: sortMstEdges -> buildDendrogramFromSortedMst -> buildCondensedTree
/// -> selectClusters -> labelPoints. If the input MST is empty (root < 0),
/// every point is labeled noise (-1) and the function returns 0.
///
/// @tparam algorithmFPType Floating-point type used for edge weights / lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in,out] mstFrom                 Source endpoint of each MST edge, length `nRows - 1` (sorted in place)
/// @param[in,out] mstTo                   Target endpoint of each MST edge, length `nRows - 1` (sorted in place)
/// @param[in,out] mstWeights              Edge weights (sort key), length `nRows - 1` (sorted in place)
/// @param[in]     nRows                   Number of original points
/// @param[in]     minClusterSize          Minimum cluster size threshold (mcs)
/// @param[out]    assignments             Output point->label table, length `nRows`
/// @param[in]     clusterSelection        0 = EOM (default), 1 = leaf
/// @param[in]     allowSingleCluster      If false, reject root-only outcomes
/// @param[in]     clusterSelectionEpsilon Distance epsilon for cluster_selection_epsilon (0 == disabled)
/// @param[in]     maxClusterSize          Maximum cluster size cap (0 == uncapped)
///
/// @return Number of distinct labels emitted (== `labelCounter`); 0 if the MST is empty
template <typename algorithmFPType, CpuType cpu>
int sortMstAndExtractClusters(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t nRows, size_t minClusterSize, int * assignments,
                              int clusterSelection = 0, bool allowSingleCluster = false, double clusterSelectionEpsilon = 0.0,
                              size_t maxClusterSize = 0)
{
    const size_t edgeCount  = nRows - 1;
    const size_t totalNodes = 2 * nRows - 1;

    sortMstEdges<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, edgeCount);

    TArray<int, cpu> nodeSizeArr(totalNodes);
    TArray<int, cpu> leftChildArr(totalNodes);
    TArray<int, cpu> rightChildArr(totalNodes);
    TArray<algorithmFPType, cpu> nodeWeightArr(totalNodes);
    int * nodeSize               = nodeSizeArr.get();
    int * leftChild              = leftChildArr.get();
    int * rightChild             = rightChildArr.get();
    algorithmFPType * nodeWeight = nodeWeightArr.get();

    const int root = buildDendrogramFromSortedMst<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, edgeCount, nodeSize, leftChild, rightChild,
                                                                        nodeWeight, totalNodes);

    if (root < 0)
    {
        for (size_t i = 0; i < nRows; i++) assignments[i] = -1;
        return 0;
    }

    int nextCid = static_cast<int>(nRows);
    TArray<int, cpu> dendroToClusterArr(totalNodes);
    int * dendroToCluster = dendroToClusterArr.get();
    for (size_t i = 0; i < totalNodes; i++) dendroToCluster[i] = -1;
    dendroToCluster[root] = nextCid++;

    const size_t maxCondensed = 3 * nRows;
    TArray<CondensedEdge, cpu> condensedArr(maxCondensed);
    TArray<algorithmFPType, cpu> condensedLambdaArr(maxCondensed);
    CondensedEdge * condensed         = condensedArr.get();
    algorithmFPType * condensedLambda = condensedLambdaArr.get();

    const int mcs     = static_cast<int>(minClusterSize);
    size_t nCondensed = buildCondensedTree<algorithmFPType, cpu>(root, nRows, mcs, nodeSize, leftChild, rightChild, nodeWeight, dendroToCluster,
                                                                 condensed, condensedLambda, nextCid);

    const int nClusters = nextCid;
    const int rootCid   = static_cast<int>(nRows);

    TArray<char, cpu> isSelectedArr(nClusters);
    char * isSelected = isSelectedArr.get();

    selectClusters<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, mcs, maxClusterSize, clusterSelection,
                                         allowSingleCluster, clusterSelectionEpsilon, isSelected);

    return labelPoints<algorithmFPType, cpu>(condensed, nCondensed, nRows, leftChild, rightChild, dendroToCluster, isSelected, nClusters, rootCid,
                                             assignments);
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_CLUSTER_UTILS_H__
