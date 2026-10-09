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

// Row, MST edge, dendrogram node and cluster ids are DAAL_INT; labels are written as int32.

/// One edge of the condensed cluster tree.
///
/// Connects a parent cluster to a child cluster or to a single point that fell out of it.
struct CondensedEdge
{
    DAAL_INT parent;    ///< Parent cluster id
    DAAL_INT child;     ///< Child cluster id or fallen-out point id (< nRows)
    DAAL_INT childSize; ///< Number of original points in the child subtree (1 for fallen leaves)
};

/// Stably sort MST edges in ascending order of weight, keeping endpoint arrays aligned.
///
/// Ties are broken by edge position, as the GPU backend's stable radix sort does, since the
/// order of tied edges decides the dendrogram. The weights must be finite.
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

    // IdxValType orders by value, then by index.
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

/// Check that Boruvka produced a complete MST with finite weights, which `sortMstEdges` needs.
///
/// @tparam algorithmFPType Floating-point type of the weights
///
/// @param[in] edgesAdded Number of edges Boruvka added
/// @param[in] edgeCount  Expected number of edges, `nRows - 1`
/// @param[in] mstWeights Edge weights, length `edgeCount`
///
/// @return `ErrorIncorrectInputNumericTable` otherwise; both happen only for non-finite input
template <typename algorithmFPType>
static services::Status checkMst(size_t edgesAdded, size_t edgeCount, const algorithmFPType * mstWeights)
{
    if (edgesAdded != edgeCount || data_management::internal::valuesAreNotFinite(mstWeights, edgeCount, false))
    {
        return services::Status(services::ErrorIncorrectInputNumericTable);
    }
    return services::Status();
}

/// Build the single-linkage dendrogram from sorted MST edges via union-find.
///
/// Leaves are the points `[0, nRows)`; MST edge `e` creates internal node `nRows + e`.
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
/// Row `e` is the merge that created node `nRows + e`: `[left, right, distance, size]`, the
/// layout of scipy's `linkage` and scikit-learn's `_single_linkage_tree_`.
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
/// A side with at least `mcs` points stays a cluster, a smaller side falls out as points at lambda
/// `1 / nodeWeight`, and a new cluster id is taken only when both sides survive.
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
    // Breadth-first, left before right, as scikit-learn's _condense_tree, which fixes the cluster
    // numbering. Each node is enqueued at most once.
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
/// Sets each child's birth lambda and size and each parent's non-leaf flag and child count.
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
/// `stability[c] = sum over edges from c of max(0, deathLambda - birthLambda) * childSize`.
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
/// Visits clusters children first. A parent whose stability beats its children's sum stays
/// selected and unselects its descendants; otherwise it takes the children's sum. Clusters larger
/// than `mcsMax` always defer to their children. The root takes part only when
/// `treeTop == rootCid`.
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
/// As scikit-learn's `epsilon_search`: the cluster is replaced by its first ancestor born above
/// epsilon, or by the root only with `allowSingleCluster`.
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
/// @param[in]     allowSingleCluster     If false, never promote up to the root
/// @param[in,out] isSelected             Selection mask updated in place, length `nClusters`
template <typename algorithmFPType, CpuType cpu>
static void applyClusterSelectionEpsilon(const CondensedEdge * condensed, size_t nCondensed, size_t nRows, DAAL_INT nClusters, DAAL_INT rootCid,
                                         const algorithmFPType * lambdaBirth, algorithmFPType clusterSelectionEpsilon, bool allowSingleCluster,
                                         bool * isSelected)
{
    TArray<DAAL_INT, cpu> clusterParentArr(nClusters);
    DAAL_INT * clusterParent = clusterParentArr.get();
    services::internal::service_memset<DAAL_INT, cpu>(clusterParent, static_cast<DAAL_INT>(-1), static_cast<size_t>(nClusters));
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<DAAL_INT>(nRows)) clusterParent[e.child] = e.parent;
    }

    // Port of scikit-learn's epsilon_search / traverse_upwards.
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
    services::internal::service_memset<DAAL_INT, cpu>(fillCursor, static_cast<DAAL_INT>(0), static_cast<size_t>(nClusters));
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
/// Leaf mode (`clusterSelection == 1`) picks every leaf cluster, EOM otherwise; then the epsilon
/// refinement when `clusterSelectionEpsilon > 0`.
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

    // Every cluster starts selected and EOM only deselects; the root only with allowSingleCluster.
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
        // Leaf mode never selects the root, as in scikit-learn, so a tree that never splits gives
        // only noise.
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
/// Relies on parents having smaller ids than their children.
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
/// Like resolveClusterLabels, but yields cluster ids, which index per-cluster arrays.
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
/// As scikit-learn's `_get_probabilities`: the lambda at which the point fell out of its cluster
/// divided by the cluster's death lambda; noise gets 0. The result is 1 when the cluster has no
/// outgoing edge, when the point's lambda is not finite, or when the point never fell out.
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
/// As scikit-learn's `_do_labelling`: a point is kept only if its drop-out lambda reaches
/// `1 / epsilon` when epsilon is set, or else the root's death lambda.
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
        // Below `1 / infLambda` the reciprocal would overflow and demote coincident points.
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

/// Assign each point the dense label of its deepest selected ancestor cluster, or -1 for noise.
///
/// Points that fell out of a cluster use that cluster; the others walk up the dendrogram to the
/// first node with a known cluster.
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
///                             are dense ids `[0, labelCounter)` or -1 for noise
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

    // At most nRows / mcs labels, which the kernel entry checks against INT32_MAX.
    int labelCounter = 0;
    TArray<int, cpu> clusterLabelArr(nClusters);
    int * clusterLabel = clusterLabelArr.get();
    services::internal::service_memset<int, cpu>(clusterLabel, -1, static_cast<size_t>(nClusters));
    for (DAAL_INT c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }
    DAAL_ASSERT(labelCounter >= 0);

    // A clustering that consists of the root alone does not separate noise by
    // itself, so the weakest members are demoted by their drop-out lambda.
    const bool singleRootSelected = allowSingleCluster && labelCounter == 1 && isSelected[rootCid];

    // Allocated only when needed: the cluster each point resolved through, and the lambda it fell
    // out at (negative if it never did).
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
    services::internal::service_memset<DAAL_INT, cpu>(clusterParent, static_cast<DAAL_INT>(-1), static_cast<size_t>(nClusters));
    services::internal::service_memset<DAAL_INT, cpu>(pointFellFrom, static_cast<DAAL_INT>(-1), nRows);
    if (needProbs)
    {
        services::internal::service_memset<DAAL_INT, cpu>(pointCluster, static_cast<DAAL_INT>(-1), nRows);
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
/// Shared by all CPU methods. An empty MST labels every point as noise.
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
    services::internal::service_memset<DAAL_INT, cpu>(dendroToCluster, static_cast<DAAL_INT>(-1), totalNodes);
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
