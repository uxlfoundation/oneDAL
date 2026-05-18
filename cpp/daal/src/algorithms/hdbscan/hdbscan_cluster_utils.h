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

struct CondensedEdge
{
    int parent;
    int child;
    int childSize;
};

template <typename algorithmFPType, CpuType cpu>
static void sortMstEdges(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t edgeCount)
{
    daal::algorithms::internal::qSort<algorithmFPType, int, int, cpu>(edgeCount, mstWeights, mstFrom, mstTo);
}

/// Build the single-linkage dendrogram from sorted MST edges via union-find.
/// Node ids [0, nRows) are leaves; [nRows, nRows+edgeCount) are internal nodes.
/// Returns the root node id, or -1 if empty.
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

static void computeChildOffsets(int nClusters, const int * childCount, int * childOffset)
{
    childOffset[0] = 0;
    for (int c = 1; c <= nClusters; c++) childOffset[c] = childOffset[c - 1] + childCount[c - 1];
}

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
/// Invariant: in the condensed tree built by buildCondensedTree, a parent's cluster id
/// is always strictly less than each child's id (parents are emitted before children),
/// so iterating c = rootCid..nClusters-1 visits parents before children and
/// resolvedLabel[clusterParent[c]] is already final by the time we read it.
/// Replaces a per-point ancestor walk with a single table lookup in labelPoints.
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
///
/// resolveClusterLabels precomputes the dense label per cluster in one
/// O(nClusters) sweep so neither per-point pass walks the cluster tree.
/// Returns the number of distinct labels emitted (0..labelCounter-1).
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
/// Shared by brute_force, kd_tree, and ball_tree methods.
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
