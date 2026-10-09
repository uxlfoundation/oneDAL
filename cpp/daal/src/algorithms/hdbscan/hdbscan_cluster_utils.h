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

#include "services/daal_defines.h"
#include "src/data_management/finiteness_checker.h"
#include "src/algorithms/service_sort.h"
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
using daal::services::internal::MaxVal;
using daal::services::internal::TArray;

// Internal index type used across the HDBSCAN CPU pipeline for row / MST-edge /
// dendrogram-node / cluster identifiers. `DAAL_INT` is 64-bit on 64-bit
// platforms, mirroring the convention used by linear_model, decision_forests,
// and other DAAL kernels that must scale beyond INT_MAX rows. Cluster ids are
// derived from `nRows + edgeIndex`, so once row indices need >32 bits every
// derived id inherits the same width. The oneAPI boundary casts these down to
// `std::int32_t` when writing the `responses` table.

/// One edge of the condensed cluster tree.
///
/// Produced by buildCondensedTree: an edge connects a parent cluster id to
/// either a child cluster id (with childSize == subtree size) or a single
/// fallen-out point id (childSize == 1).
struct CondensedEdge
{
    DAAL_INT parent;    ///< Parent cluster id
    DAAL_INT child;     ///< Child cluster id or fallen-out point id (< nRows)
    DAAL_INT childSize; ///< Number of original points in the child subtree (1 for fallen leaves)
};

/// Stably sort MST edges in ascending order of weight, keeping endpoint arrays aligned.
///
/// Required by buildDendrogramFromSortedMst, which folds the edges in this order, so among tied
/// edges the order decides tie-degenerate splits -- and equal weights are the norm under MRD. The
/// sort is therefore on `(weight, position)` pairs, matching the GPU backend's stable radix sort,
/// with a fallback to the in-place `qSort` if the buffers cannot be allocated, which preserves
/// ascending weights and leaves only the tie order unspecified. The weights must be finite.
///
/// @tparam algorithmFPType Floating-point type used for edge weights
/// @tparam cpu             CPU dispatch tag
///
/// @param[in,out] mstFrom    Source endpoint of each MST edge, length `edgeCount`
/// @param[in,out] mstTo      Target endpoint of each MST edge, length `edgeCount`
/// @param[in,out] mstWeights Edge weights (sort key), length `edgeCount`
/// @param[in]     edgeCount  Number of MST edges
template <typename algorithmFPType, CpuType cpu>
static void sortMstEdges(DAAL_INT * mstFrom, DAAL_INT * mstTo, algorithmFPType * mstWeights, size_t edgeCount)
{
    TArray<daal::IdxValType<algorithmFPType>, cpu> pairsArr(edgeCount);
    TArray<DAAL_INT, cpu> fromArr(edgeCount);
    TArray<DAAL_INT, cpu> toArr(edgeCount);
    daal::IdxValType<algorithmFPType> * pairs = pairsArr.get();
    DAAL_INT * sortedFrom                     = fromArr.get();
    DAAL_INT * sortedTo                       = toArr.get();

    if (pairs == nullptr || sortedFrom == nullptr || sortedTo == nullptr)
    {
        daal::algorithms::internal::qSort<algorithmFPType, DAAL_INT, DAAL_INT, cpu>(edgeCount, mstWeights, mstFrom, mstTo);
        return;
    }

    for (size_t i = 0; i < edgeCount; i++)
    {
        pairs[i].value = mstWeights[i];
        pairs[i].index = i;
    }

    // Orders by value, then by index: the (weight, position) order.
    daal::parallel_sort<algorithmFPType>(pairs, pairs + edgeCount);

    for (size_t i = 0; i < edgeCount; i++)
    {
        const size_t src = pairs[i].index;
        sortedFrom[i]    = mstFrom[src];
        sortedTo[i]      = mstTo[src];
        mstWeights[i]    = pairs[i].value;
    }

    const size_t idxBytes = edgeCount * sizeof(DAAL_INT);
    daal::services::internal::daal_memcpy_s(mstFrom, idxBytes, sortedFrom, idxBytes);
    daal::services::internal::daal_memcpy_s(mstTo, idxBytes, sortedTo, idxBytes);
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
static DAAL_INT buildDendrogramFromSortedMst(const DAAL_INT * mstFrom, const DAAL_INT * mstTo, const algorithmFPType * mstWeights, size_t nRows,
                                             size_t edgeCount, DAAL_INT * nodeSize, DAAL_INT * leftChild, DAAL_INT * rightChild,
                                             algorithmFPType * nodeWeight, size_t totalNodes)
{
    TArray<DAAL_INT, cpu> ufParentArr(nRows);
    TArray<DAAL_INT, cpu> compSizeArr(nRows);
    TArray<DAAL_INT, cpu> compToNodeArr(nRows);
    DAAL_INT * ufParent   = ufParentArr.get();
    DAAL_INT * compSize   = compSizeArr.get();
    DAAL_INT * compToNode = compToNodeArr.get();

    // Both init loops write at the `TArray` base pointer via `service_memset_seq`
    // which internally aligns its stores; the SIMD annotation on the first loop
    // covers each of the four arrays' base offset. For the second loop the
    // stores go through `service_memset_seq` too so no `aligned(...)` clause
    // is needed on a SIMD loop -- the helper is the memset dispatch path shared
    // with other DAAL kernels.
    PRAGMA_OMP_SIMD_ARGS(aligned(nodeSize, leftChild, rightChild, nodeWeight : DAAL_MALLOC_DEFAULT_ALIGNMENT))
    for (size_t i = 0; i < nRows; i++)
    {
        nodeSize[i]   = 1;
        leftChild[i]  = -1;
        rightChild[i] = -1;
        nodeWeight[i] = algorithmFPType(0);
    }
    if (totalNodes > nRows)
    {
        const size_t tail = totalNodes - nRows;
        services::internal::service_memset_seq<DAAL_INT, cpu>(nodeSize + nRows, static_cast<DAAL_INT>(0), tail);
        services::internal::service_memset_seq<DAAL_INT, cpu>(leftChild + nRows, static_cast<DAAL_INT>(-1), tail);
        services::internal::service_memset_seq<DAAL_INT, cpu>(rightChild + nRows, static_cast<DAAL_INT>(-1), tail);
        services::internal::service_memset_seq<algorithmFPType, cpu>(nodeWeight + nRows, algorithmFPType(0), tail);
    }
    // Union-find init: writes only at offset 0 of every `TArray` base.
    PRAGMA_OMP_SIMD_ARGS(aligned(ufParent, compSize, compToNode : DAAL_MALLOC_DEFAULT_ALIGNMENT))
    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]   = static_cast<DAAL_INT>(i);
        compSize[i]   = 1;
        compToNode[i] = static_cast<DAAL_INT>(i);
    }

    auto ufFind = [&](DAAL_INT x) -> DAAL_INT {
        while (ufParent[x] != x)
        {
            ufParent[x] = ufParent[ufParent[x]];
            x           = ufParent[x];
        }
        return x;
    };

    DAAL_INT root = -1;
    for (size_t e = 0; e < edgeCount; e++)
    {
        const DAAL_INT ru = ufFind(mstFrom[e]);
        const DAAL_INT rv = ufFind(mstTo[e]);
        if (ru == rv) continue;

        const DAAL_INT nodeId  = static_cast<DAAL_INT>(nRows + e);
        const DAAL_INT newSize = compSize[ru] + compSize[rv];

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

/// Dump the single-linkage dendrogram as a row-major `(nRows - 1) x 4` matrix.
///
/// Row `e` describes the merge that created internal node `nRows + e`:
/// `[leftChild, rightChild, mergeDistance, subtreeSize]`. This is the layout
/// scipy's `linkage` and scikit-learn's `_single_linkage_tree_` use, so a caller
/// can re-cut the hierarchy at an arbitrary distance -- what
/// `HDBSCAN.dbscan_clustering` does -- without rebuilding it.
///
/// Node ids need no remapping: `buildDendrogramFromSortedMst` keys internal node
/// `e` off the MST edge index, and every MST edge joins two distinct components
/// (the edge set is a forest, so no edge is ever redundant whatever order it is
/// processed in), which makes ids `[nRows, nRows + edgeCount)` dense.
///
/// @tparam algorithmFPType Floating-point type used for merge distances
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  nRows      Number of original points
/// @param[in]  edgeCount  Number of merges (`nRows - 1`)
/// @param[in]  nodeSize   Subtree size per node, length `2*nRows - 1`
/// @param[in]  leftChild  Left child id per node, length `2*nRows - 1`
/// @param[in]  rightChild Right child id per node, length `2*nRows - 1`
/// @param[in]  nodeWeight Merge distance per node, length `2*nRows - 1`
/// @param[out] tree       Output matrix, length `4 * edgeCount`
template <typename algorithmFPType, CpuType cpu>
static void dumpSingleLinkageTree(size_t nRows, size_t edgeCount, const DAAL_INT * nodeSize, const DAAL_INT * leftChild, const DAAL_INT * rightChild,
                                  const algorithmFPType * nodeWeight, algorithmFPType * tree)
{
    for (size_t e = 0; e < edgeCount; e++)
    {
        const size_t nid    = nRows + e;
        algorithmFPType * r = tree + 4 * e;
        r[0]                = static_cast<algorithmFPType>(leftChild[nid]);
        r[1]                = static_cast<algorithmFPType>(rightChild[nid]);
        r[2]                = nodeWeight[nid];
        r[3]                = static_cast<algorithmFPType>(nodeSize[nid]);
    }
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
static size_t buildCondensedTree(DAAL_INT root, size_t nRows, DAAL_INT mcs, const DAAL_INT * nodeSize, const DAAL_INT * leftChild,
                                 const DAAL_INT * rightChild, const algorithmFPType * nodeWeight, DAAL_INT * dendroToCluster,
                                 CondensedEdge * condensed, algorithmFPType * condensedLambda, DAAL_INT & nextCid)
{
    TArray<DAAL_INT, cpu> leafStackArr(nRows);
    TArray<DAAL_INT, cpu> fallenBufArr(nRows);
    DAAL_INT * leafStack = leafStackArr.get();
    DAAL_INT * fallenBuf = fallenBufArr.get();

    auto collectLeaves = [&](DAAL_INT startNid, DAAL_INT * out, size_t & outCount) {
        outCount              = 0;
        size_t stackTop       = 0;
        leafStack[stackTop++] = startNid;
        while (stackTop > 0)
        {
            const DAAL_INT nid = leafStack[--stackTop];
            if (nid < static_cast<DAAL_INT>(nRows))
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
        DAAL_INT node;
        DAAL_INT cluster;
    };
    // A FIFO queue, so the nodes are visited breadth-first, left before right, as
    // scikit-learn's _condense_tree does; that order is what numbers the clusters.
    // Each dendrogram node is enqueued at most once, so 2*nRows slots suffice.
    TArray<StackItem, cpu> mainStackArr(2 * nRows);
    StackItem * mainStack     = mainStackArr.get();
    size_t mainStackHead      = 0;
    size_t mainStackTop       = 0;
    mainStack[mainStackTop++] = { root, dendroToCluster[root] };

    size_t nCondensed = 0;

    auto emitFallenLeaves = [&](DAAL_INT subtree, DAAL_INT parentCid, algorithmFPType lambda) {
        size_t nFallen = 0;
        collectLeaves(subtree, fallenBuf, nFallen);
        for (size_t fi = 0; fi < nFallen; fi++)
        {
            condensed[nCondensed]       = { parentCid, fallenBuf[fi], 1 };
            condensedLambda[nCondensed] = lambda;
            nCondensed++;
        }
    };

    while (mainStackHead < mainStackTop)
    {
        const StackItem item     = mainStack[mainStackHead++];
        const DAAL_INT nid       = item.node;
        const DAAL_INT parentCid = item.cluster;

        if (nid < static_cast<DAAL_INT>(nRows)) continue;

        const DAAL_INT lc = leftChild[nid];
        const DAAL_INT rc = rightChild[nid];
        if (lc < 0 || rc < 0) continue;

        const DAAL_INT ls            = nodeSize[lc];
        const DAAL_INT rs            = nodeSize[rc];
        const algorithmFPType lambda = (nodeWeight[nid] > algorithmFPType(0)) ? algorithmFPType(1) / nodeWeight[nid] : MaxVal<algorithmFPType>::get();

        const bool lBig = ls >= mcs;
        const bool rBig = rs >= mcs;

        if (lBig && rBig)
        {
            const DAAL_INT lcid         = nextCid++;
            const DAAL_INT rcid         = nextCid++;
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
/// @param[out] isLeafCluster   true if cluster has no cluster-children, false otherwise; length `nClusters`
/// @param[out] clusterSz       Number of points in each cluster, length `nClusters`
/// @param[out] childCount      Number of cluster-children per cluster, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void initClusterMetadata(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                                DAAL_INT nClusters, DAAL_INT rootCid, algorithmFPType * lambdaBirth, bool * isLeafCluster, DAAL_INT * clusterSz,
                                DAAL_INT * childCount)
{
    for (DAAL_INT c = 0; c < nClusters; c++)
    {
        lambdaBirth[c]   = algorithmFPType(0);
        isLeafCluster[c] = true;
        clusterSz[c]     = 0;
        childCount[c]    = 0;
    }
    clusterSz[rootCid] = static_cast<DAAL_INT>(nRows);

    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<DAAL_INT>(nRows))
        {
            lambdaBirth[e.child]    = condensedLambda[ei];
            isLeafCluster[e.parent] = false;
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
static void computeClusterStability(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, DAAL_INT nClusters,
                                    const algorithmFPType * lambdaBirth, algorithmFPType * stability)
{
    services::internal::service_memset_seq<algorithmFPType, cpu>(stability, algorithmFPType(0), static_cast<size_t>(nClusters));
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
/// Iterates clusters in decreasing id order (children before parents) down to
/// `treeTop` (inclusive). For each non-leaf cluster, compares the parent's
/// stability against the sum of its children's stabilities. If children win,
/// the parent is unselected and its stability is replaced by the sum (so its
/// grandparent sees the propagated score). If the parent wins, every
/// descendant is unselected via an explicit stack walk over
/// `childOffset`/`childList`. Oversized clusters (size > `mcsMax`) are forced
/// onto the children-win branch unconditionally, leaf clusters included.
///
/// `treeTop` controls whether the root cluster participates. When the caller
/// allows a single-cluster outcome, `treeTop == rootCid` and the root may win
/// EOM. Otherwise, `treeTop == rootCid + 1` and the root is never visited;
/// it must be deselected up front by the caller.
///
/// @tparam algorithmFPType Floating-point type used for cluster stabilities
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     nClusters     Total cluster count
/// @param[in]     treeTop       Lowest cluster id to visit (rootCid or rootCid+1)
/// @param[in]     mcsMax        Maximum allowed cluster size (`MaxVal<DAAL_INT>` if uncapped)
/// @param[in,out] stability     Per-cluster stability; updated in place with propagated child sums
/// @param[in]     clusterSz     Per-cluster point counts, length `nClusters`
/// @param[in]     isLeafCluster Leaf-cluster mask (`bool`), length `nClusters`
/// @param[in]     childOffset   CSR offsets into `childList`, length `nClusters + 1`
/// @param[in]     childCount    Per-cluster cluster-child counts, length `nClusters`
/// @param[in]     childList     CSR child cluster ids, length `childOffset[nClusters]`
/// @param[in,out] descStack     Scratch stack for the descendant-unselect walk, length `nClusters`
/// @param[in,out] isSelected    Selection mask updated in place, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void runEomSelection(DAAL_INT nClusters, DAAL_INT treeTop, DAAL_INT mcsMax, algorithmFPType * stability, const DAAL_INT * clusterSz,
                            const bool * isLeafCluster, const DAAL_INT * childOffset, const DAAL_INT * childCount, const DAAL_INT * childList,
                            DAAL_INT * descStack, bool * isSelected)
{
    for (DAAL_INT c = nClusters - 1; c >= treeTop; c--)
    {
        if (isLeafCluster[c])
        {
            // No children to compare against, so only the size cap can unselect a leaf; its
            // propagated stability is the empty child sum, as on the children-win branch.
            if (clusterSz[c] > mcsMax)
            {
                isSelected[c] = false;
                stability[c]  = algorithmFPType(0);
            }
            continue;
        }

        algorithmFPType childSum       = algorithmFPType(0);
        const DAAL_INT childOffsetC    = childOffset[c];
        const DAAL_INT childOffsetCEnd = childOffsetC + childCount[c];
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : childSum))
        for (DAAL_INT ci = childOffsetC; ci < childOffsetCEnd; ci++) childSum += stability[childList[ci]];

        const bool oversized = (clusterSz[c] > mcsMax);

        if (oversized || childSum > stability[c])
        {
            isSelected[c] = false;
            stability[c]  = childSum;
        }
        else
        {
            size_t descTop = 0;
            for (DAAL_INT ci = childOffset[c]; ci < childOffset[c] + childCount[c]; ci++) descStack[descTop++] = childList[ci];
            while (descTop > 0)
            {
                const DAAL_INT d = descStack[--descTop];
                isSelected[d]    = false;
                for (DAAL_INT ci = childOffset[d]; ci < childOffset[d] + childCount[d]; ci++) descStack[descTop++] = childList[ci];
            }
        }
    }
}

/// Promote selected clusters that are too dense (birth distance < epsilon) to their parent.
///
/// Implements the cluster_selection_epsilon refinement exactly as scikit-learn's
/// `epsilon_search`: a selected cluster born below `clusterSelectionEpsilon` is
/// replaced by its first ancestor born above it, or by the root when
/// `allowSingleCluster` is set, or kept when only the root lies above.
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
/// @param[in]     clusterSelectionEpsilon Distance threshold; clusters with birth distance below this are merged into parent.
///                                        Passed in as `algorithmFPType` so the inner comparison against
///                                        `birthDist` avoids any implicit `float`->`double` promotion inside the tight loop.
/// @param[in]     allowSingleCluster     If false, never promote up to the root
/// @param[in,out] isSelected             Selection mask updated in place, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void applyClusterSelectionEpsilon(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, DAAL_INT nClusters, DAAL_INT rootCid,
                                         const algorithmFPType * lambdaBirth, algorithmFPType clusterSelectionEpsilon, bool allowSingleCluster,
                                         bool * isSelected)
{
    TArray<DAAL_INT, cpu> clusterParentArr(nClusters);
    DAAL_INT * clusterParent = clusterParentArr.get();
    services::internal::service_memset<DAAL_INT, cpu>(clusterParent, DAAL_INT(-1), static_cast<size_t>(nClusters));
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<DAAL_INT>(nRows)) clusterParent[e.child] = e.parent;
    }

    // A literal port of scikit-learn's epsilon_search / traverse_upwards. A selected cluster born
    // below epsilon climbs while its parent was born at or below epsilon, stopping at the first
    // parent born above it; the root is reached only with allowSingleCluster. Clusters under an
    // earlier target are skipped, while those born at or above epsilon stay selected as they are.
    if (isSelected[rootCid]) return;
    auto birthDist = [&](DAAL_INT c) -> algorithmFPType {
        return (lambdaBirth[c] > algorithmFPType(0)) ? algorithmFPType(1) / lambdaBirth[c] : algorithmFPType(0);
    };
    TArray<bool, cpu> pickedArr(nClusters);
    TArray<bool, cpu> isTargetArr(nClusters);
    bool * picked   = pickedArr.get();
    bool * isTarget = isTargetArr.get();
    if (!picked || !isTarget) return;
    services::internal::service_memset<bool, cpu>(picked, false, static_cast<size_t>(nClusters));
    services::internal::service_memset<bool, cpu>(isTarget, false, static_cast<size_t>(nClusters));

    for (DAAL_INT c = rootCid + 1; c < nClusters; c++)
    {
        if (!isSelected[c]) continue;
        if (!(birthDist(c) < clusterSelectionEpsilon))
        {
            picked[c] = true;
            continue;
        }
        bool processed = false;
        for (DAAL_INT a = clusterParent[c]; a >= rootCid && a < nClusters; a = clusterParent[a])
        {
            if (isTarget[a])
            {
                processed = true;
                break;
            }
            if (a == rootCid) break;
        }
        if (processed) continue;

        DAAL_INT target = c;
        while (true)
        {
            const DAAL_INT parent = clusterParent[target];
            if (parent < rootCid || parent >= nClusters) break;
            if (parent == rootCid)
            {
                if (allowSingleCluster) target = rootCid;
                break;
            }
            target = parent;
            if (birthDist(parent) > clusterSelectionEpsilon) break;
        }
        picked[target]   = true;
        isTarget[target] = true;
    }
    for (DAAL_INT c = 0; c < nClusters; c++) isSelected[c] = picked[c];
}

/// Build CSR-style child offsets via prefix-sum over per-cluster child counts.
///
/// @param[in]  nClusters   Total cluster count
/// @param[in]  childCount  Per-cluster cluster-child counts, length `nClusters`
/// @param[out] childOffset Prefix sums; `childOffset[c]` is the start of c's children, length `nClusters + 1`
static void computeChildOffsets(DAAL_INT nClusters, const DAAL_INT * childCount, DAAL_INT * childOffset)
{
    childOffset[0] = 0;
    for (DAAL_INT c = 1; c <= nClusters; c++) childOffset[c] = childOffset[c - 1] + childCount[c - 1];
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
static void fillChildList(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, DAAL_INT nClusters, const DAAL_INT * childOffset,
                          DAAL_INT * childList)
{
    TArray<DAAL_INT, cpu> fillCursorArr(nClusters);
    DAAL_INT * fillCursor = fillCursorArr.get();
    services::internal::service_memset<DAAL_INT, cpu>(fillCursor, DAAL_INT(0), static_cast<size_t>(nClusters));
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<DAAL_INT>(nRows))
        {
            childList[childOffset[e.parent] + fillCursor[e.parent]] = e.child;
            fillCursor[e.parent]++;
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
/// followed by applyClusterSelectionEpsilon (if `clusterSelectionEpsilon > 0`).
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
static void selectClusters(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                           DAAL_INT nClusters, DAAL_INT rootCid, DAAL_INT mcs, size_t maxClusterSize, int clusterSelection, bool allowSingleCluster,
                           double clusterSelectionEpsilon, bool * isSelected)
{
    TArray<algorithmFPType, cpu> stabilityArr(nClusters);
    TArray<algorithmFPType, cpu> lambdaBirthArr(nClusters);
    // Predicate-only mask: `bool` is the natural type; DAAL exposes
    // `TArray<bool, cpu>` (see e.g. df_classification_predict, svm_train_boser)
    // for the same use case.
    TArray<bool, cpu> isLeafClusterArr(nClusters);
    TArray<DAAL_INT, cpu> clusterSzArr(nClusters);
    TArray<DAAL_INT, cpu> childCountArr(nClusters);
    algorithmFPType * stability   = stabilityArr.get();
    algorithmFPType * lambdaBirth = lambdaBirthArr.get();
    bool * isLeafCluster          = isLeafClusterArr.get();
    DAAL_INT * clusterSz          = clusterSzArr.get();
    DAAL_INT * childCount         = childCountArr.get();

    initClusterMetadata<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, lambdaBirth, isLeafCluster,
                                              clusterSz, childCount);

    TArray<DAAL_INT, cpu> childOffsetArr(nClusters + 1);
    DAAL_INT * childOffset = childOffsetArr.get();
    computeChildOffsets(nClusters, childCount, childOffset);
    const DAAL_INT totalChildren = childOffset[nClusters];

    TArray<DAAL_INT, cpu> childListArr(totalChildren > 0 ? totalChildren : 1);
    DAAL_INT * childList = childListArr.get();
    fillChildList<algorithmFPType, cpu>(condensed, nCondensed, nRows, nClusters, childOffset, childList);

    computeClusterStability<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nClusters, lambdaBirth, stability);

    const DAAL_INT mcsMax = (maxClusterSize > 0) ? static_cast<DAAL_INT>(maxClusterSize) : MaxVal<DAAL_INT>::get();

    // Seed isSelected: every internal cluster id starts selected and EOM only
    // deselects. Leaf-mode resets the mask before picking leaves directly.
    // The root is included only when allowSingleCluster permits it; otherwise
    // EOM never visits it and it stays deselected.
    for (DAAL_INT c = 0; c < nClusters; c++)
    {
        isSelected[c] = false;
    }
    for (DAAL_INT c = rootCid + 1; c < nClusters; c++)
    {
        if (clusterSz[c] >= mcs) isSelected[c] = true;
    }
    if (allowSingleCluster && clusterSz[rootCid] >= mcs) isSelected[rootCid] = true;

    if (clusterSelection == 1)
    {
        // Leaf mode picks the leaves of the *cluster* tree, whose nodes are the
        // child clusters only, so the root is never a candidate -- not even when
        // allowSingleCluster is set, which in leaf mode only relaxes the labeling
        // threshold. A condensed tree that never splits therefore offers no
        // candidate at all and every point ends up as noise.
        isSelected[rootCid] = false;
        for (DAAL_INT c = rootCid + 1; c < nClusters; c++)
        {
            isSelected[c] = (isLeafCluster[c] && clusterSz[c] >= mcs);
        }
    }
    else
    {
        const DAAL_INT treeTop = allowSingleCluster ? rootCid : (rootCid + 1);
        if (allowSingleCluster)
        {
            // scikit-learn caps the root by the points its child clusters hold, not by every point.
            DAAL_INT rootSize = 0;
            for (DAAL_INT k = childOffset[rootCid]; k < childOffset[rootCid] + childCount[rootCid]; k++) rootSize += clusterSz[childList[k]];
            clusterSz[rootCid] = rootSize;
        }
        TArray<DAAL_INT, cpu> descStackArr(nClusters);
        DAAL_INT * descStack = descStackArr.get();
        runEomSelection<algorithmFPType, cpu>(nClusters, treeTop, mcsMax, stability, clusterSz, isLeafCluster, childOffset, childCount, childList,
                                              descStack, isSelected);
    }

    if (clusterSelectionEpsilon > 0.0)
    {
        applyClusterSelectionEpsilon<algorithmFPType, cpu>(condensed, nCondensed, nRows, nClusters, rootCid, lambdaBirth,
                                                           static_cast<algorithmFPType>(clusterSelectionEpsilon), allowSingleCluster, isSelected);
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
static void resolveClusterLabels(DAAL_INT rootCid, DAAL_INT nClusters, const bool * isSelected, const int * clusterLabel,
                                 const DAAL_INT * clusterParent, int * resolvedLabel)
{
    for (DAAL_INT c = 0; c < rootCid; c++) resolvedLabel[c] = -1;
    for (DAAL_INT c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c])
        {
            resolvedLabel[c] = clusterLabel[c];
            continue;
        }
        const DAAL_INT p = clusterParent[c];
        resolvedLabel[c] = (p >= rootCid && p < nClusters) ? resolvedLabel[p] : -1;
    }
}

/// Resolve, for every cluster, the id of the deepest selected ancestor (itself included).
///
/// Same forward sweep and same parent-before-child invariant as
/// resolveClusterLabels; it resolves to the cluster *id* rather than to the
/// dense label because the membership probability has to index per-cluster
/// arrays (the death lambda) and the dense label is not such an index.
///
/// @param[in]  rootCid         Root cluster id (== `nRows`)
/// @param[in]  nClusters       Total cluster count
/// @param[in]  isSelected      Selection mask, length `nClusters`
/// @param[in]  clusterParent   Parent cluster id per cluster (-1 for root), length `nClusters`
/// @param[out] resolvedCluster Deepest selected ancestor per cluster (-1 if none), length `nClusters`
static void resolveSelectedAncestors(DAAL_INT rootCid, DAAL_INT nClusters, const bool * isSelected, const DAAL_INT * clusterParent,
                                     DAAL_INT * resolvedCluster)
{
    for (DAAL_INT c = 0; c < rootCid; c++) resolvedCluster[c] = -1;
    for (DAAL_INT c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c])
        {
            resolvedCluster[c] = c;
            continue;
        }
        const DAAL_INT p   = clusterParent[c];
        resolvedCluster[c] = (p >= rootCid && p < nClusters) ? resolvedCluster[p] : -1;
    }
}

/// Membership strength of every point in the cluster it was assigned to.
///
/// Mirrors scikit-learn's `_hdbscan.hdbscan._get_probabilities`: a point's
/// strength is the lambda at which it dropped out of the condensed tree,
/// normalized by the lambda at which its cluster died (the largest lambda over
/// the cluster's outgoing condensed edges). 1 means the point survived to the
/// very end of its cluster, values near 0 mean it detached almost immediately,
/// and noise is 0.
///
/// Three cases give exactly 1:
///   - the cluster has no outgoing edge, so its death lambda is 0 and the
///     normalization is undefined (sklearn's `max_lambda == 0.0` branch);
///   - the point's own lambda is not finite, which happens for duplicate points
///     at zero mutual reachability distance (sklearn's `not isfinite` branch --
///     this implementation substitutes the largest finite value instead of an
///     infinity, which the clamp below handles identically);
///   - the point never dropped out before its cluster died, i.e. it is one of
///     the points that labelPoints has to resolve through the dendrogram; such
///     a point outlived the cluster by definition.
///
/// @tparam algorithmFPType Floating-point type used for lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed       Condensed-tree edges, length `nCondensed`
/// @param[in]  condensedLambda Death lambda per edge, length `nCondensed`
/// @param[in]  nCondensed      Number of condensed-tree edges
/// @param[in]  nRows           Number of original points
/// @param[in]  nClusters       Total cluster count
/// @param[in]  rootCid         Root cluster id (== `nRows`)
/// @param[in]  pointCluster    Condensed cluster each point was resolved through (-1 if noise), length `nRows`
/// @param[in]  pointLambda     Lambda at which each point dropped out, or a negative
///                             value if it never did, length `nRows`
/// @param[in]  resolvedCluster Deepest selected ancestor per cluster, length `nClusters`
/// @param[in]  assignments     Final point labels (-1 for noise), length `nRows`
/// @param[out] probabilities   Output membership strength in `[0, 1]`, length `nRows`
template <typename algorithmFPType, CpuType cpu>
static void computeMembershipProbabilities(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                                           DAAL_INT nClusters, DAAL_INT rootCid, const DAAL_INT * pointCluster, const algorithmFPType * pointLambda,
                                           const DAAL_INT * resolvedCluster, const int * assignments, algorithmFPType * probabilities)
{
    TArray<algorithmFPType, cpu> clusterDeathArr(nClusters);
    algorithmFPType * clusterDeath = clusterDeathArr.get();
    services::internal::service_memset<algorithmFPType, cpu>(clusterDeath, algorithmFPType(0), static_cast<size_t>(nClusters));
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const DAAL_INT p = condensed[ei].parent;
        if (p >= rootCid && p < nClusters && condensedLambda[ei] > clusterDeath[p]) clusterDeath[p] = condensedLambda[ei];
    }

    for (size_t i = 0; i < nRows; i++)
    {
        probabilities[i] = algorithmFPType(0);
        if (assignments[i] < 0) continue;

        const DAAL_INT cid = pointCluster[i];
        if (cid < rootCid || cid >= nClusters) continue;
        const DAAL_INT sc = resolvedCluster[cid];
        if (sc < rootCid || sc >= nClusters) continue;

        const algorithmFPType maxLambda = clusterDeath[sc];
        const algorithmFPType lambda    = pointLambda[i];
        if (!(maxLambda > algorithmFPType(0)) || lambda < algorithmFPType(0))
        {
            probabilities[i] = algorithmFPType(1);
            continue;
        }
        probabilities[i] = (lambda < maxLambda) ? lambda / maxLambda : algorithmFPType(1);
    }
}

/// Reverse the child arrays into a parent pointer for every dendrogram node.
///
/// @param[in]  leftChild    Left child id for every internal node, length `totalNodes`
/// @param[in]  rightChild   Right child id for every internal node, length `totalNodes`
/// @param[in]  nRows        Number of leaves (internal nodes start at index `nRows`)
/// @param[in]  totalNodes   Total node count (`2 * nRows - 1`)
/// @param[out] dendroParent Parent id per node (-1 for root and unused slots), length `totalNodes`
static void buildDendroParent(const DAAL_INT * leftChild, const DAAL_INT * rightChild, size_t nRows, size_t totalNodes, DAAL_INT * dendroParent)
{
    for (size_t i = 0; i < totalNodes; i++) dendroParent[i] = -1;
    for (size_t nid = nRows; nid < totalNodes; nid++)
    {
        if (leftChild[nid] >= 0) dendroParent[leftChild[nid]] = static_cast<DAAL_INT>(nid);
        if (rightChild[nid] >= 0) dendroParent[rightChild[nid]] = static_cast<DAAL_INT>(nid);
    }
}

/// Demote the weakest members of a root-only clustering to noise.
///
/// When the only selected cluster is the root, the flat clustering has no
/// sibling to separate noise from signal, so scikit-learn's `_do_labelling`
/// keeps a point only if it survived long enough: its drop-out lambda must reach
/// a threshold. With `cluster_selection_epsilon` set, the threshold is
/// `1 / epsilon` -- the lambda that corresponds to that distance. Otherwise it
/// is the root's own death lambda, i.e. the largest lambda over every edge
/// leaving the root, which keeps only the points that made it to the very end.
///
/// A point that never dropped out has `pointLambda < 0` and is demoted, which is
/// the conservative reading of "did not reach the threshold"; buildCondensedTree
/// emits every point exactly once for `mcs >= 2`, so this cannot happen in
/// practice.
///
/// @tparam algorithmFPType Floating-point type used for lambdas
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]     condensed               Condensed-tree edges, length `nCondensed`
/// @param[in]     condensedLambda         Death lambda per edge, length `nCondensed`
/// @param[in]     nCondensed              Number of condensed-tree edges
/// @param[in]     nRows                   Number of original points
/// @param[in]     rootCid                 Root cluster id (== `nRows`)
/// @param[in]     rootLabel               Dense label assigned to the root cluster
/// @param[in]     pointLambda             Drop-out lambda per point (negative if it never dropped out), length `nRows`
/// @param[in]     clusterSelectionEpsilon Distance epsilon (0 == disabled)
/// @param[in,out] assignments             Point->label table, length `nRows`
///
/// @return true if at least one point kept the root label
template <typename algorithmFPType, CpuType cpu>
static bool applySingleClusterThreshold(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                                        DAAL_INT rootCid, int rootLabel, const algorithmFPType * pointLambda, double clusterSelectionEpsilon,
                                        int * assignments)
{
    const algorithmFPType infLambda = MaxVal<algorithmFPType>::get();

    algorithmFPType threshold = algorithmFPType(0);
    if (clusterSelectionEpsilon > 0.0)
    {
        // Below `1 / infLambda` the reciprocal overflows, in double or in the
        // narrowing, which would put the threshold above the zero-distance
        // lambda and demote the coincident points an infinite lambda keeps.
        const double minEpsilon = 1.0 / static_cast<double>(infLambda);
        threshold               = (clusterSelectionEpsilon > minEpsilon) ? static_cast<algorithmFPType>(1.0 / clusterSelectionEpsilon) : infLambda;
    }
    else
    {
        for (size_t ei = 0; ei < nCondensed; ei++)
        {
            if (condensed[ei].parent == rootCid && condensedLambda[ei] > threshold) threshold = condensedLambda[ei];
        }
    }

    bool anyKept = false;
    for (size_t i = 0; i < nRows; i++)
    {
        if (assignments[i] != rootLabel) continue;
        if (pointLambda[i] >= threshold)
            anyKept = true;
        else
            assignments[i] = -1;
    }
    return anyKept;
}

/// Final point labeling phase. For each input point i, assign assignments[i] to
///   - the dense label (0..nLabels-1) of the deepest selected ancestor cluster
///     in the condensed tree, OR
///   - -1 if no such ancestor exists (the point is "noise").
///
/// Two sequential passes over the points:
///   1) Points that fell out of a cluster directly (pointFellFrom[i] >= 0):
///      look up the resolved label of their drop cluster in O(1).
///   2) Points that never fell out before the root cluster: walk up the
///      dendrogram via dendroParent[] until hitting a node with a known
///      cluster id, then look up its resolved label.
/// resolveClusterLabels precomputes the dense label per cluster in one
/// O(nClusters) sweep so neither per-point pass walks the cluster tree.
/// A third pass runs only when the clustering ends up being the root cluster
/// alone, see applySingleClusterThreshold.
///
/// @tparam algorithmFPType Floating-point type of the lambdas and probabilities
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  condensed       Condensed-tree edges, length `nCondensed`
/// @param[in]  condensedLambda Death lambda per edge, length `nCondensed`
/// @param[in]  nCondensed      Number of condensed-tree edges
/// @param[in]  nRows           Number of original points
/// @param[in]  leftChild       Left child id per dendrogram node, length `2*nRows - 1`
/// @param[in]  rightChild      Right child id per dendrogram node, length `2*nRows - 1`
/// @param[in]  dendroToCluster Maps dendrogram node id -> cluster id (-1 if no cluster), length `2*nRows - 1`
/// @param[in]  isSelected      Selection mask, length `nClusters`
/// @param[in]  nClusters       Total cluster count
/// @param[in]  rootCid         Root cluster id (== `nRows`)
/// @param[in]  allowSingleCluster      Whether the root alone may form the clustering
/// @param[in]  clusterSelectionEpsilon Distance epsilon (0 == disabled); drives the
///                                     single-cluster labeling threshold
/// @param[out] assignments     Output point->label table, length `nRows`. Labels
///                             are dense ids `[0, labelCounter)` or -1 for
///                             noise, both fit into `int` regardless of
///                             `nRows`, so the write happens with an explicit
///                             narrowing cast at this layer.
/// @param[out] probabilities   Optional output membership strength in `[0, 1]`,
///                             length `nRows`. Pass `nullptr` to skip the
///                             computation and its scratch allocations
///
/// @return Number of distinct labels emitted (0..labelCounter-1); 0 when the
///         single-cluster threshold demoted every point to noise
template <typename algorithmFPType, CpuType cpu>
static int labelPoints(const CondensedEdge * condensed, const algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                       const DAAL_INT * leftChild, const DAAL_INT * rightChild, const DAAL_INT * dendroToCluster, const bool * isSelected,
                       DAAL_INT nClusters, DAAL_INT rootCid, bool allowSingleCluster, double clusterSelectionEpsilon, int * assignments,
                       algorithmFPType * probabilities = nullptr)
{
    const size_t totalNodes = 2 * nRows - 1;

    // labelCounter fits in `int`: total distinct labels is bounded by the number
    // of selected clusters, itself bounded by `nRows / mcs` (`mcs >= 2` is
    // enforced at the kernel entry). The DAAL kernel entry point refuses
    // inputs with `nRows / minClusterSize > INT32_MAX` up front (see the guard
    // at the top of each `_batch_impl.i::compute`), so this routine can
    // materialise labels as `int` (and `-1` for noise) directly. The
    // `int`-typed labels contract is codebase-wide across DAAL (kmeans, knn,
    // decision_forest, svm, etc. all use `WriteOnlyRows<int, cpu>`); widening
    // it here alone would break the convention.
    int labelCounter = 0;
    TArray<int, cpu> clusterLabelArr(nClusters);
    int * clusterLabel = clusterLabelArr.get();
    services::internal::service_memset<int, cpu>(clusterLabel, -1, static_cast<size_t>(nClusters));
    for (DAAL_INT c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }
    // Defensive: unreachable for realistic inputs, but the invariant should
    // fire loudly if some future change (e.g. mcs = 1) breaks it.
    DAAL_ASSERT(labelCounter >= 0);

    // A clustering that consists of the root alone does not separate noise by
    // itself, so the weakest members are demoted by their drop-out lambda.
    const bool singleRootSelected = allowSingleCluster && labelCounter == 1 && isSelected[rootCid];

    // The scratch arrays are only allocated when something below needs them;
    // `pointCluster` records which condensed cluster each point was resolved
    // through (the dense label alone cannot index per-cluster data) and
    // `pointLambda` the lambda it dropped out at, negative if it never did.
    const bool needProbs       = (probabilities != nullptr);
    const bool needPointLambda = needProbs || singleRootSelected;

    TArray<DAAL_INT, cpu> clusterParentArr(nClusters);
    TArray<DAAL_INT, cpu> pointFellFromArr(nRows);
    TArray<DAAL_INT, cpu> pointClusterArr(needProbs ? nRows : 0);
    TArray<algorithmFPType, cpu> pointLambdaArr(needPointLambda ? nRows : 0);
    DAAL_INT * clusterParent      = clusterParentArr.get();
    DAAL_INT * pointFellFrom      = pointFellFromArr.get();
    DAAL_INT * pointCluster       = pointClusterArr.get();
    algorithmFPType * pointLambda = pointLambdaArr.get();
    services::internal::service_memset<DAAL_INT, cpu>(clusterParent, DAAL_INT(-1), static_cast<size_t>(nClusters));
    services::internal::service_memset<DAAL_INT, cpu>(pointFellFrom, DAAL_INT(-1), nRows);
    if (needProbs)
    {
        services::internal::service_memset<DAAL_INT, cpu>(pointCluster, DAAL_INT(-1), nRows);
    }
    if (needPointLambda)
    {
        services::internal::service_memset<algorithmFPType, cpu>(pointLambda, algorithmFPType(-1), nRows);
    }
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<DAAL_INT>(nRows))
            clusterParent[e.child] = e.parent;
        else
        {
            pointFellFrom[e.child] = e.parent;
            if (needPointLambda) pointLambda[e.child] = condensedLambda[ei];
        }
    }

    TArray<int, cpu> resolvedLabelArr(nClusters);
    int * resolvedLabel = resolvedLabelArr.get();
    resolveClusterLabels(rootCid, nClusters, isSelected, clusterLabel, clusterParent, resolvedLabel);

    for (size_t i = 0; i < nRows; i++)
    {
        const DAAL_INT c = pointFellFrom[i];
        assignments[i]   = (c >= rootCid && c < nClusters) ? resolvedLabel[c] : -1;
        if (needProbs) pointCluster[i] = c;
    }

    TArray<DAAL_INT, cpu> dendroParentArr(totalNodes);
    DAAL_INT * dendroParent = dendroParentArr.get();
    buildDendroParent(leftChild, rightChild, nRows, totalNodes, dendroParent);

    for (size_t i = 0; i < nRows; i++)
    {
        if (pointFellFrom[i] >= 0) continue;

        DAAL_INT nid = static_cast<DAAL_INT>(i);
        while (nid >= 0 && nid < static_cast<DAAL_INT>(totalNodes))
        {
            const DAAL_INT cid = dendroToCluster[nid];
            if (cid >= rootCid)
            {
                assignments[i] = resolvedLabel[cid];
                if (needProbs) pointCluster[i] = cid;
                break;
            }
            nid = dendroParent[nid];
        }
    }

    if (singleRootSelected
        && !applySingleClusterThreshold<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, rootCid, clusterLabel[rootCid],
                                                              pointLambda, clusterSelectionEpsilon, assignments))
    {
        // Every point was demoted, so the root label is unused and the result has
        // no cluster at all.
        labelCounter = 0;
    }

    if (needProbs)
    {
        TArray<DAAL_INT, cpu> resolvedClusterArr(nClusters);
        DAAL_INT * resolvedCluster = resolvedClusterArr.get();
        resolveSelectedAncestors(rootCid, nClusters, isSelected, clusterParent, resolvedCluster);
        computeMembershipProbabilities<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, pointCluster,
                                                             pointLambda, resolvedCluster, assignments, probabilities);
    }

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
/// @param[out]    probabilities           Optional membership strength per point in `[0, 1]`,
///                                        length `nRows`. Pass `nullptr` to skip it
/// @param[out]    singleLinkageTree       Optional row-major `(nRows - 1) x 4` dendrogram dump,
///                                        `[left, right, distance, size]` per merge. Pass `nullptr`
///                                        to skip it
///
/// @return Number of distinct labels emitted (== `labelCounter`); 0 if there are no points at
///         all, if the MST is empty, or if every point ended up as noise
template <typename algorithmFPType, CpuType cpu>
int sortMstAndExtractClusters(DAAL_INT * mstFrom, DAAL_INT * mstTo, algorithmFPType * mstWeights, size_t nRows, size_t minClusterSize,
                              int * assignments, int clusterSelection = 0, bool allowSingleCluster = false, double clusterSelectionEpsilon = 0.0,
                              size_t maxClusterSize = 0, algorithmFPType * probabilities = nullptr, algorithmFPType * singleLinkageTree = nullptr)
{
    // `edgeCount` and `totalNodes` below wrap for `nRows == 0` and then become allocation lengths.
    if (nRows == 0)
    {
        return 0;
    }
    const size_t edgeCount  = nRows - 1;
    const size_t totalNodes = 2 * nRows - 1;

    sortMstEdges<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, edgeCount);

    TArray<DAAL_INT, cpu> nodeSizeArr(totalNodes);
    TArray<DAAL_INT, cpu> leftChildArr(totalNodes);
    TArray<DAAL_INT, cpu> rightChildArr(totalNodes);
    TArray<algorithmFPType, cpu> nodeWeightArr(totalNodes);
    DAAL_INT * nodeSize          = nodeSizeArr.get();
    DAAL_INT * leftChild         = leftChildArr.get();
    DAAL_INT * rightChild        = rightChildArr.get();
    algorithmFPType * nodeWeight = nodeWeightArr.get();

    const DAAL_INT root = buildDendrogramFromSortedMst<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, edgeCount, nodeSize, leftChild,
                                                                             rightChild, nodeWeight, totalNodes);

    if (root < 0)
    {
        services::internal::service_memset<int, cpu>(assignments, -1, nRows);
        if (probabilities != nullptr)
        {
            services::internal::service_memset<algorithmFPType, cpu>(probabilities, algorithmFPType(0), nRows);
        }
        if (singleLinkageTree != nullptr)
        {
            services::internal::service_memset<algorithmFPType, cpu>(singleLinkageTree, algorithmFPType(0), 4 * edgeCount);
        }
        return 0;
    }

    if (singleLinkageTree != nullptr)
    {
        dumpSingleLinkageTree<algorithmFPType, cpu>(nRows, edgeCount, nodeSize, leftChild, rightChild, nodeWeight, singleLinkageTree);
    }

    DAAL_INT nextCid = static_cast<DAAL_INT>(nRows);
    TArray<DAAL_INT, cpu> dendroToClusterArr(totalNodes);
    DAAL_INT * dendroToCluster = dendroToClusterArr.get();
    services::internal::service_memset<DAAL_INT, cpu>(dendroToCluster, DAAL_INT(-1), totalNodes);
    dendroToCluster[root] = nextCid++;

    const size_t maxCondensed = 3 * nRows;
    TArray<CondensedEdge, cpu> condensedArr(maxCondensed);
    TArray<algorithmFPType, cpu> condensedLambdaArr(maxCondensed);
    CondensedEdge * condensed         = condensedArr.get();
    algorithmFPType * condensedLambda = condensedLambdaArr.get();

    const DAAL_INT mcs = static_cast<DAAL_INT>(minClusterSize);
    size_t nCondensed  = buildCondensedTree<algorithmFPType, cpu>(root, nRows, mcs, nodeSize, leftChild, rightChild, nodeWeight, dendroToCluster,
                                                                  condensed, condensedLambda, nextCid);

    const DAAL_INT nClusters = nextCid;
    const DAAL_INT rootCid   = static_cast<DAAL_INT>(nRows);

    TArray<bool, cpu> isSelectedArr(nClusters);
    bool * isSelected = isSelectedArr.get();

    selectClusters<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, mcs, maxClusterSize, clusterSelection,
                                         allowSingleCluster, clusterSelectionEpsilon, isSelected);

    return labelPoints<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, leftChild, rightChild, dendroToCluster, isSelected,
                                             nClusters, rootCid, allowSingleCluster, clusterSelectionEpsilon, assignments, probabilities);
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_CLUSTER_UTILS_H__
