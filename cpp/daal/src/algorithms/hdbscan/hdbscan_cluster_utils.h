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
#include <vector>

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

struct CondensedEdge
{
    int parent;
    int child;
    int childSize;
};

template <typename algorithmFPType, CpuType cpu>
static void sortMstEdges(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t edgeCount)
{
    struct MstEdge
    {
        algorithmFPType weight;
        int from;
        int to;
    };

    std::vector<MstEdge> edges(edgeCount);

    for (size_t i = 0; i < edgeCount; i++)
    {
        edges[i] = { mstWeights[i], mstFrom[i], mstTo[i] };
    }

    std::sort(edges.begin(), edges.end(), [](const MstEdge & a, const MstEdge & b) {
        if (a.weight != b.weight) return a.weight < b.weight;
        const int aLo = (a.from < a.to) ? a.from : a.to;
        const int aHi = (a.from < a.to) ? a.to : a.from;
        const int bLo = (b.from < b.to) ? b.from : b.to;
        const int bHi = (b.from < b.to) ? b.to : b.from;
        if (aLo != bLo) return aLo < bLo;
        return aHi < bHi;
    });

    for (size_t i = 0; i < edgeCount; i++)
    {
        mstFrom[i]    = edges[i].from;
        mstTo[i]      = edges[i].to;
        mstWeights[i] = edges[i].weight;
    }
}

template <typename algorithmFPType, CpuType cpu>
struct DendroNode
{
    int left;
    int right;
    algorithmFPType weight;
    int size;
};

template <typename algorithmFPType, CpuType cpu>
static int buildKruskalDendrogram(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t nRows, size_t edgeCount,
                                  DendroNode<algorithmFPType, cpu> * dendro, int * nodeSize, int * leftChild, int * rightChild,
                                  algorithmFPType * nodeWeight, size_t totalNodes)
{
    std::vector<int> ufParent(nRows);
    std::vector<int> compSize(nRows, 1);
    std::vector<int> compToNode(nRows);

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]   = static_cast<int>(i);
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

    for (size_t i = 0; i < edgeCount; i++) dendro[i] = { 0, 0, algorithmFPType(0), 0 };

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

    // Initialize node arrays
    for (size_t i = 0; i < totalNodes; i++)
    {
        nodeSize[i]   = (i < nRows) ? 1 : 0;
        leftChild[i]  = -1;
        rightChild[i] = -1;
        nodeWeight[i] = algorithmFPType(0);
    }
    for (size_t e = 0; e < edgeCount; e++)
    {
        const size_t nid = nRows + e;
        nodeSize[nid]    = dendro[e].size;
        leftChild[nid]   = dendro[e].left;
        rightChild[nid]  = dendro[e].right;
        nodeWeight[nid]  = dendro[e].weight;
    }

    // Find root
    int root = -1;
    for (int e = static_cast<int>(edgeCount) - 1; e >= 0; e--)
    {
        if (dendro[e].size > 0)
        {
            root = static_cast<int>(nRows + e);
            break;
        }
    }
    return root;
}

template <typename algorithmFPType, CpuType cpu>
static size_t buildCondensedTree(int root, size_t nRows, int mcs, int * nodeSize, int * leftChild, int * rightChild, algorithmFPType * nodeWeight,
                                 int * dendroToCluster, CondensedEdge * condensed, algorithmFPType * condensedLambda, int & nextCid)
{
    std::vector<int> leafStack(nRows);
    std::vector<int> fallenBuf(nRows);

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
    std::vector<StackItem> mainStack(nDendroNodes + 1);
    int mainStackTop          = 0;
    mainStack[mainStackTop++] = { root, dendroToCluster[root] };

    size_t nCondensed = 0;

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
            size_t nFallen      = 0;
            collectLeaves(rc, fallenBuf.data(), nFallen);
            for (size_t fi = 0; fi < nFallen; fi++)
            {
                condensed[nCondensed]       = { parentCid, fallenBuf[fi], 1 };
                condensedLambda[nCondensed] = lambda;
                nCondensed++;
            }
            mainStack[mainStackTop++] = { lc, parentCid };
        }
        else if (rBig)
        {
            dendroToCluster[rc] = parentCid;
            size_t nFallen      = 0;
            collectLeaves(lc, fallenBuf.data(), nFallen);
            for (size_t fi = 0; fi < nFallen; fi++)
            {
                condensed[nCondensed]       = { parentCid, fallenBuf[fi], 1 };
                condensedLambda[nCondensed] = lambda;
                nCondensed++;
            }
            mainStack[mainStackTop++] = { rc, parentCid };
        }
        else
        {
            size_t nFallen = 0;
            collectLeaves(lc, fallenBuf.data(), nFallen);
            size_t nFallen2 = 0;
            collectLeaves(rc, fallenBuf.data() + nFallen, nFallen2);
            nFallen += nFallen2;
            for (size_t fi = 0; fi < nFallen; fi++)
            {
                condensed[nCondensed]       = { parentCid, fallenBuf[fi], 1 };
                condensedLambda[nCondensed] = lambda;
                nCondensed++;
            }
        }
    }

    return nCondensed;
}

template <typename algorithmFPType, CpuType cpu>
static void selectClusters(CondensedEdge * condensed, algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows, int nClusters, int rootCid,
                           int mcs, size_t maxClusterSize, int clusterSelection, bool allowSingleCluster, double clusterSelectionEpsilon,
                           char * isSelected)
{
    std::vector<algorithmFPType> stability(nClusters, algorithmFPType(0));
    std::vector<algorithmFPType> lambdaBirth(nClusters, algorithmFPType(0));
    std::vector<char> isLeafCluster(nClusters, 1);
    std::vector<int> clusterSz(nClusters, 0);
    clusterSz[rootCid] = static_cast<int>(nRows);

    std::vector<int> childCount(nClusters, 0);

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

    std::vector<int> childOffset(nClusters + 1);
    childOffset[0] = 0;
    for (int c = 1; c <= nClusters; c++) childOffset[c] = childOffset[c - 1] + childCount[c - 1];
    const int totalChildren = childOffset[nClusters];

    std::vector<int> childList(totalChildren > 0 ? totalChildren : 1);

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
        const algorithmFPType contrib = (condensedLambda[ei] - birth) * static_cast<algorithmFPType>(e.childSize);
        if (contrib > algorithmFPType(0)) stability[e.parent] += contrib;
    }

    const int mcsMax = (maxClusterSize > 0) ? static_cast<int>(maxClusterSize) : std::numeric_limits<int>::max();

    for (int c = 0; c < nClusters; c++)
    {
        isSelected[c] = (c >= rootCid && clusterSz[c] >= mcs) ? 1 : 0;
    }

    std::vector<int> descStack(nClusters);

    if (clusterSelection == 1)
    {
        for (int c = rootCid; c < nClusters; c++)
        {
            isSelected[c] = (isLeafCluster[c] && clusterSz[c] >= mcs) ? 1 : 0;
        }
    }
    else
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

    // allow_single_cluster enforcement
    if (!allowSingleCluster)
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

    // cluster_selection_epsilon merging
    if (clusterSelectionEpsilon > 0.0)
    {
        std::vector<int> clusterParent(nClusters, -1);
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
}

template <typename algorithmFPType, CpuType cpu>
static int labelPoints(CondensedEdge * condensed, algorithmFPType * condensedLambda, size_t nCondensed, size_t nRows,
                       DendroNode<algorithmFPType, cpu> * dendro, int * dendroToCluster, char * isSelected, int nClusters, int rootCid,
                       int * assignments)
{
    const size_t totalNodes   = 2 * nRows - 1;
    const size_t nDendroNodes = nRows - 1;

    int labelCounter = 0;
    std::vector<int> clusterLabel(nClusters, -1);
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }

    // Build cluster parent map for label walk-up
    std::vector<int> clusterParent(nClusters, -1);
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child >= static_cast<int>(nRows)) clusterParent[e.child] = e.parent;
    }

    std::vector<int> pointFellFrom(nRows, -1);
    for (size_t ei = 0; ei < nCondensed; ei++)
    {
        const CondensedEdge & e = condensed[ei];
        if (e.child < static_cast<int>(nRows)) pointFellFrom[e.child] = e.parent;
    }

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

    // Handle points never ejected
    std::vector<int> dendroParent(totalNodes, -1);
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

/// Sort MST edges by weight and extract flat clusters via condensed tree + EOM.
/// Shared by both brute_force and kd_tree methods.
template <typename algorithmFPType, CpuType cpu>
int sortMstAndExtractClusters(int * mstFrom, int * mstTo, algorithmFPType * mstWeights, size_t nRows, size_t minClusterSize, int * assignments,
                              int clusterSelection = 0, bool allowSingleCluster = false, double clusterSelectionEpsilon = 0.0,
                              size_t maxClusterSize = 0)
{
    const size_t edgeCount  = nRows - 1;
    const size_t totalNodes = 2 * nRows - 1;

    // Step 1: Sort MST edges
    sortMstEdges<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, edgeCount);

    // Step 2: Build dendrogram
    std::vector<DendroNode<algorithmFPType, cpu> > dendroVec(edgeCount);
    std::vector<int> nodeSizeVec(totalNodes);
    std::vector<int> leftChildVec(totalNodes);
    std::vector<int> rightChildVec(totalNodes);
    std::vector<algorithmFPType> nodeWeightVec(totalNodes);
    auto * dendro                = dendroVec.data();
    int * nodeSize               = nodeSizeVec.data();
    int * leftChild              = leftChildVec.data();
    int * rightChild             = rightChildVec.data();
    algorithmFPType * nodeWeight = nodeWeightVec.data();

    int root = buildKruskalDendrogram<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, edgeCount, dendro, nodeSize, leftChild, rightChild,
                                                            nodeWeight, totalNodes);

    if (root < 0)
    {
        for (size_t i = 0; i < nRows; i++) assignments[i] = -1;
        return 0;
    }

    // Step 3: Build condensed tree
    int nextCid = static_cast<int>(nRows);
    std::vector<int> dendroToClusterVec(totalNodes, -1);
    int * dendroToCluster = dendroToClusterVec.data();
    dendroToCluster[root] = nextCid++;

    const size_t maxCondensed = 3 * nRows;
    std::vector<CondensedEdge> condensedVec(maxCondensed);
    std::vector<algorithmFPType> condensedLambdaVec(maxCondensed);
    CondensedEdge * condensed         = condensedVec.data();
    algorithmFPType * condensedLambda = condensedLambdaVec.data();

    const int mcs     = static_cast<int>(minClusterSize);
    size_t nCondensed = buildCondensedTree<algorithmFPType, cpu>(root, nRows, mcs, nodeSize, leftChild, rightChild, nodeWeight, dendroToCluster,
                                                                 condensed, condensedLambda, nextCid);

    // Step 4: Select clusters
    const int nClusters = nextCid;
    const int rootCid   = static_cast<int>(nRows);

    std::vector<char> isSelectedVec(nClusters);
    char * isSelected = isSelectedVec.data();

    selectClusters<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, nClusters, rootCid, mcs, maxClusterSize, clusterSelection,
                                         allowSingleCluster, clusterSelectionEpsilon, isSelected);

    // Step 5: Label points
    return labelPoints<algorithmFPType, cpu>(condensed, condensedLambda, nCondensed, nRows, dendro, dendroToCluster, isSelected, nClusters, rootCid,
                                             assignments);
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_CLUSTER_UTILS_H__
