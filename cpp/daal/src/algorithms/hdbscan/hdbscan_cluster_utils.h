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

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

using daal::internal::CpuType;

/// Sort MST edges by weight and extract flat clusters via condensed tree + EOM.
/// Shared by both brute_force and kd_tree methods.
///
/// Input:  mstFrom[edgeCount], mstTo[edgeCount], mstWeights[edgeCount] — unsorted MST
/// Output: assignments[nRows] — cluster labels (-1 = noise), returns label count
template <typename algorithmFPType, CpuType cpu>
int sortMstAndExtractClusters(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t nRows, size_t minClusterSize, int * assignments)
{
    const size_t edgeCount    = nRows - 1;
    const size_t nDendroNodes = nRows - 1;
    const size_t totalNodes   = 2 * nRows - 1;

    // =========================================================================
    // Step 4: Sort MST edges by weight
    // =========================================================================

    struct MstEdge
    {
        algorithmFPType weight;
        int from;
        int to;
    };

    {
        daal::services::internal::TArray<MstEdge, cpu> edgesArr(edgeCount);
        MstEdge * edges = edgesArr.get();
        if (!edges) return 0;

        for (size_t i = 0; i < edgeCount; i++)
        {
            edges[i] = { mstWeights[i], mstFrom[i], mstTo[i] };
        }

        std::sort(edges, edges + edgeCount, [](const MstEdge & a, const MstEdge & b) { return a.weight < b.weight; });

        for (size_t i = 0; i < edgeCount; i++)
        {
            mstFrom[i]    = edges[i].from;
            mstTo[i]      = edges[i].to;
            mstWeights[i] = edges[i].weight;
        }
    }

    // =========================================================================
    // Step 5: Extract clusters (dendrogram -> condensed tree -> EOM -> labels)
    // =========================================================================

    // Phase 1: Build Kruskal dendrogram via union-find
    daal::services::internal::TArray<int, cpu> ufParentArr(nRows);
    daal::services::internal::TArray<int, cpu> compSizeArr(nRows);
    int * ufParent = ufParentArr.get();
    int * compSize = compSizeArr.get();
    if (!ufParent || !compSize) return 0;
    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i] = static_cast<int>(i);
        compSize[i] = 1;
    }

    auto ufFind = [&](int x) -> int {
        while (ufParent[x] != x)
        {
            ufParent[x] = ufParent[ufParent[x]];
            x           = ufParent[x];
        }
        return x;
    };

    struct DendroNode
    {
        int left;
        int right;
        algorithmFPType weight;
        int size;
    };

    daal::services::internal::TArray<DendroNode, cpu> dendroArr(nDendroNodes);
    DendroNode * dendro = dendroArr.get();
    if (!dendro) return 0;
    for (size_t i = 0; i < nDendroNodes; i++)
    {
        dendro[i] = { 0, 0, algorithmFPType(0), 0 };
    }

    daal::services::internal::TArray<int, cpu> compToNodeArr(nRows);
    int * compToNode = compToNodeArr.get();
    if (!compToNode) return 0;
    for (size_t i = 0; i < nRows; i++)
    {
        compToNode[i] = static_cast<int>(i);
    }

    for (size_t e = 0; e < edgeCount; e++)
    {
        const int ru = ufFind(mstFrom[e]);
        const int rv = ufFind(mstTo[e]);
        if (ru == rv) continue;

        const int leftNode  = compToNode[ru];
        const int rightNode = compToNode[rv];
        const int newSize   = compSize[ru] + compSize[rv];
        const int nodeId    = static_cast<int>(nRows + e);

        dendro[e] = { leftNode, rightNode, mstWeights[e], newSize };

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

    // Phase 2: Build condensed tree
    daal::services::internal::TArray<int, cpu> nodeSizeArr(totalNodes);
    daal::services::internal::TArray<int, cpu> leftChildArr(totalNodes);
    daal::services::internal::TArray<int, cpu> rightChildArr(totalNodes);
    daal::services::internal::TArray<algorithmFPType, cpu> nodeWeightArr(totalNodes);
    int * nodeSize               = nodeSizeArr.get();
    int * leftChild              = leftChildArr.get();
    int * rightChild             = rightChildArr.get();
    algorithmFPType * nodeWeight = nodeWeightArr.get();
    if (!nodeSize || !leftChild || !rightChild || !nodeWeight) return 0;

    for (size_t i = 0; i < totalNodes; i++)
    {
        nodeSize[i]   = (i < nRows) ? 1 : 0;
        leftChild[i]  = -1;
        rightChild[i] = -1;
        nodeWeight[i] = algorithmFPType(0);
    }
    for (size_t e = 0; e < nDendroNodes; e++)
    {
        const size_t nid = nRows + e;
        nodeSize[nid]    = dendro[e].size;
        leftChild[nid]   = dendro[e].left;
        rightChild[nid]  = dendro[e].right;
        nodeWeight[nid]  = dendro[e].weight;
    }

    struct CondensedEdge
    {
        int parent;
        int child;
        algorithmFPType lambdaVal;
        int childSize;
    };

    const size_t maxCondensed = 2 * nRows;
    daal::services::internal::TArray<CondensedEdge, cpu> condensedArr(maxCondensed);
    CondensedEdge * condensed = condensedArr.get();
    if (!condensed) return 0;
    size_t nCondensed = 0;

    // Find root
    int root = -1;
    for (int e = static_cast<int>(nDendroNodes) - 1; e >= 0; e--)
    {
        if (dendro[e].size > 0)
        {
            root = static_cast<int>(nRows + e);
            break;
        }
    }

    if (root < 0)
    {
        for (size_t i = 0; i < nRows; i++) assignments[i] = -1;
        return 0;
    }

    int nextCid = static_cast<int>(nRows);
    daal::services::internal::TArray<int, cpu> dendroToClusterArr(totalNodes);
    int * dendroToCluster = dendroToClusterArr.get();
    if (!dendroToCluster) return 0;
    for (size_t i = 0; i < totalNodes; i++) dendroToCluster[i] = -1;
    dendroToCluster[root] = nextCid++;

    // Pre-allocated stacks
    daal::services::internal::TArray<int, cpu> leafStackArr(nDendroNodes + 1);
    int * leafStack = leafStackArr.get();
    if (!leafStack) return 0;

    daal::services::internal::TArray<int, cpu> fallenBufArr(nRows);
    int * fallenBuf = fallenBufArr.get();
    if (!fallenBuf) return 0;

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
    daal::services::internal::TArray<StackItem, cpu> mainStackArr(nDendroNodes + 1);
    StackItem * mainStack = mainStackArr.get();
    if (!mainStack) return 0;
    int mainStackTop          = 0;
    mainStack[mainStackTop++] = { root, dendroToCluster[root] };

    const int mcs = static_cast<int>(minClusterSize);

    while (mainStackTop > 0)
    {
        const StackItem item = mainStack[--mainStackTop];
        const int nid        = item.node;
        const int parentCid  = item.cluster;

        if (nid < static_cast<int>(nRows)) continue;

        const int lc = leftChild[nid];
        const int rc = rightChild[nid];
        if (lc < 0 || rc < 0) continue;

        const int ls = nodeSize[lc];
        const int rs = nodeSize[rc];
        const algorithmFPType lambda =
            (nodeWeight[nid] > algorithmFPType(0)) ? algorithmFPType(1) / nodeWeight[nid] : std::numeric_limits<algorithmFPType>::max();

        const bool lBig = ls >= mcs;
        const bool rBig = rs >= mcs;

        if (lBig && rBig)
        {
            const int lcid            = nextCid++;
            const int rcid            = nextCid++;
            dendroToCluster[lc]       = lcid;
            dendroToCluster[rc]       = rcid;
            condensed[nCondensed++]   = { parentCid, lcid, lambda, ls };
            condensed[nCondensed++]   = { parentCid, rcid, lambda, rs };
            mainStack[mainStackTop++] = { lc, lcid };
            mainStack[mainStackTop++] = { rc, rcid };
        }
        else if (lBig)
        {
            dendroToCluster[lc] = parentCid;
            size_t nFallen      = 0;
            collectLeaves(rc, fallenBuf, nFallen);
            for (size_t fi = 0; fi < nFallen; fi++) condensed[nCondensed++] = { parentCid, fallenBuf[fi], lambda, 1 };
            mainStack[mainStackTop++] = { lc, parentCid };
        }
        else if (rBig)
        {
            dendroToCluster[rc] = parentCid;
            size_t nFallen      = 0;
            collectLeaves(lc, fallenBuf, nFallen);
            for (size_t fi = 0; fi < nFallen; fi++) condensed[nCondensed++] = { parentCid, fallenBuf[fi], lambda, 1 };
            mainStack[mainStackTop++] = { rc, parentCid };
        }
        else
        {
            size_t nFallen = 0;
            collectLeaves(lc, fallenBuf, nFallen);
            size_t nFallen2 = 0;
            collectLeaves(rc, fallenBuf + nFallen, nFallen2);
            nFallen += nFallen2;
            for (size_t fi = 0; fi < nFallen; fi++) condensed[nCondensed++] = { parentCid, fallenBuf[fi], lambda, 1 };
        }
    }

    // Phase 3: Stability and EOM selection
    const int nClusters = nextCid;
    const int rootCid   = static_cast<int>(nRows);

    daal::services::internal::TArrayCalloc<algorithmFPType, cpu> stabilityArr(nClusters);
    daal::services::internal::TArrayCalloc<algorithmFPType, cpu> lambdaBirthArr(nClusters);
    daal::services::internal::TArray<char, cpu> isLeafClusterArr(nClusters);
    daal::services::internal::TArrayCalloc<int, cpu> clusterSizeArr(nClusters);
    algorithmFPType * stability   = stabilityArr.get();
    algorithmFPType * lambdaBirth = lambdaBirthArr.get();
    char * isLeafCluster          = isLeafClusterArr.get();
    int * clusterSz               = clusterSizeArr.get();
    if (!stability || !lambdaBirth || !isLeafCluster || !clusterSz) return 0;

    for (int c = 0; c < nClusters; c++) isLeafCluster[c] = 1;
    clusterSz[rootCid] = static_cast<int>(nRows);

    daal::services::internal::TArrayCalloc<int, cpu> childCountArr(nClusters);
    int * childCount = childCountArr.get();
    if (!childCount) return 0;

    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows))
        {
            lambdaBirth[e.child]    = e.lambdaVal;
            isLeafCluster[e.parent] = 0;
            childCount[e.parent]++;
            clusterSz[e.child] = e.childSize;
        }
    }

    daal::services::internal::TArray<int, cpu> childOffsetArr(nClusters + 1);
    int * childOffset = childOffsetArr.get();
    if (!childOffset) return 0;
    childOffset[0] = 0;
    for (int c = 1; c <= nClusters; c++) childOffset[c] = childOffset[c - 1] + childCount[c - 1];
    const int totalChildren = childOffset[nClusters];

    daal::services::internal::TArray<int, cpu> childListArr(totalChildren > 0 ? totalChildren : 1);
    int * childList = childListArr.get();
    if (!childList) return 0;

    for (int c = 0; c < nClusters; c++) childCount[c] = 0;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows))
        {
            childList[childOffset[e.parent] + childCount[e.parent]] = e.child;
            childCount[e.parent]++;
        }
    }

    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e       = condensed[ei];
        const algorithmFPType birth   = lambdaBirth[e.parent];
        const algorithmFPType contrib = (e.lambdaVal - birth) * static_cast<algorithmFPType>(e.childSize);
        if (contrib > algorithmFPType(0)) stability[e.parent] += contrib;
    }

    daal::services::internal::TArray<char, cpu> isSelectedArr(nClusters);
    char * isSelected = isSelectedArr.get();
    if (!isSelected) return 0;
    for (int c = 0; c < nClusters; c++)
    {
        isSelected[c] = (c >= rootCid && clusterSz[c] >= mcs) ? 1 : 0;
    }

    for (int c = nClusters - 1; c >= rootCid; c--)
    {
        if (isLeafCluster[c]) continue;

        algorithmFPType childSum = algorithmFPType(0);
        for (int ci = childOffset[c]; ci < childOffset[c] + childCount[c]; ci++) childSum += stability[childList[ci]];

        if (childSum > stability[c])
        {
            isSelected[c] = 0;
            stability[c]  = childSum;
        }
        else
        {
            int descTop = 0;
            for (int ci = childOffset[c]; ci < childOffset[c] + childCount[c]; ci++) leafStack[descTop++] = childList[ci];
            while (descTop > 0)
            {
                const int d   = leafStack[--descTop];
                isSelected[d] = 0;
                for (int ci = childOffset[d]; ci < childOffset[d] + childCount[d]; ci++) leafStack[descTop++] = childList[ci];
            }
        }
    }

    // Phase 3b: allow_single_cluster=false (matches sklearn default).
    // If only the root cluster is selected AND it has children, force-select
    // its children instead. If the root is a leaf (no children), keep it.
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

    // Phase 4: Label points
    int labelCounter = 0;
    daal::services::internal::TArray<int, cpu> clusterLabelArr(nClusters);
    int * clusterLabel = clusterLabelArr.get();
    if (!clusterLabel) return 0;
    for (int c = 0; c < nClusters; c++) clusterLabel[c] = -1;
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }

    daal::services::internal::TArray<int, cpu> clusterParentArr(nClusters);
    int * clusterParent = clusterParentArr.get();
    if (!clusterParent) return 0;
    for (int c = 0; c < nClusters; c++) clusterParent[c] = -1;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows)) clusterParent[e.child] = e.parent;
    }

    daal::services::internal::TArray<int, cpu> pointFellFromArr(nRows);
    int * pointFellFrom = pointFellFromArr.get();
    if (!pointFellFrom) return 0;
    for (size_t i = 0; i < nRows; i++) pointFellFrom[i] = -1;
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child < static_cast<int>(nRows)) pointFellFrom[e.child] = e.parent;
    }

    // Label points that fell from clusters in the condensed tree.
    // Walk up from the fell-from cluster to find the deepest selected ancestor.
    for (size_t i = 0; i < nRows; i++)
    {
        assignments[i] = -1;
        int c          = pointFellFrom[i];
        if (c < rootCid || c >= nClusters) continue;

        while (c >= rootCid && c < nClusters)
        {
            if (isSelected[c])
            {
                assignments[i] = clusterLabel[c];
                break;
            }
            c = clusterParent[c];
        }
    }

    // Handle points never ejected (remained in leaf clusters through condensation).
    // Walk up the dendrogram to find their condensed cluster, then find the
    // nearest selected ancestor.
    daal::services::internal::TArray<int, cpu> dendroParentArr(totalNodes);
    int * dendroParent = dendroParentArr.get();
    if (!dendroParent) return 0;
    for (size_t i = 0; i < totalNodes; i++) dendroParent[i] = -1;
    for (size_t e = 0; e < nDendroNodes; e++)
    {
        const size_t nid = nRows + e;
        if (dendro[e].size > 0)
        {
            dendroParent[dendro[e].left]  = static_cast<int>(nid);
            dendroParent[dendro[e].right] = static_cast<int>(nid);
        }
    }

    for (size_t i = 0; i < nRows; i++)
    {
        if (pointFellFrom[i] >= 0) continue;

        int nid = static_cast<int>(i);
        while (nid >= 0 && nid < static_cast<int>(totalNodes))
        {
            const int cid = dendroToCluster[nid];
            if (cid >= rootCid)
            {
                int c = cid;
                while (c >= rootCid && c < nClusters)
                {
                    if (isSelected[c])
                    {
                        assignments[i] = clusterLabel[c];
                        break;
                    }
                    c = clusterParent[c];
                }
                break;
            }
            nid = dendroParent[nid];
        }
    }

    return labelCounter;
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_CLUSTER_UTILS_H__
