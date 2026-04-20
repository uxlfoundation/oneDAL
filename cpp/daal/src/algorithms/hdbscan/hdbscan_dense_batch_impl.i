/* file: hdbscan_dense_batch_impl.i */
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"
#include "src/threading/threading.h"
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
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;
using daal::internal::BlasInst;

template <typename algorithmFPType, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                   NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples)
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

    // Read all input data
    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // =========================================================================
    // Step 1: Compute pairwise Euclidean distance matrix using GEMM
    // dist²[i,j] = ||a_i||² + ||a_j||² - 2 * (A * A^T)[i,j]
    // =========================================================================

    // Allocate distance matrix
    daal::services::internal::TArray<algorithmFPType, cpu> distMatrixArr(nRows * nRows);
    algorithmFPType * distMatrix = distMatrixArr.get();
    DAAL_CHECK_MALLOC(distMatrix);

    // Compute row norms: norm[i] = sum(data[i,:]^2)
    daal::services::internal::TArray<algorithmFPType, cpu> normsArr(nRows);
    algorithmFPType * norms = normsArr.get();
    DAAL_CHECK_MALLOC(norms);

    const size_t normBlockSize = 512;
    const size_t nNormBlocks   = nRows / normBlockSize + (nRows % normBlockSize > 0);

    daal::threader_for(nNormBlocks, nNormBlocks, [&](size_t iBlock) {
        const size_t begin = iBlock * normBlockSize;
        const size_t end   = (begin + normBlockSize > nRows) ? nRows : begin + normBlockSize;

        for (size_t i = begin; i < end; i++)
        {
            algorithmFPType sum         = algorithmFPType(0);
            const algorithmFPType * row = data + i * nCols;
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t d = 0; d < nCols; d++)
            {
                sum += row[d] * row[d];
            }
            norms[i] = sum;
        }
    });

    // Compute A * A^T via GEMM: dist_matrix = data * data^T
    {
        const char transa           = 't';
        const char transb           = 'n';
        const DAAL_INT m            = static_cast<DAAL_INT>(nRows);
        const DAAL_INT n            = static_cast<DAAL_INT>(nRows);
        const DAAL_INT k            = static_cast<DAAL_INT>(nCols);
        const algorithmFPType alpha = algorithmFPType(1);
        const algorithmFPType beta  = algorithmFPType(0);
        const DAAL_INT lda          = static_cast<DAAL_INT>(nCols);
        const DAAL_INT ldb          = static_cast<DAAL_INT>(nCols);
        const DAAL_INT ldc          = static_cast<DAAL_INT>(nRows);

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &n, &k, &alpha, data, &lda, data, &ldb, &beta, distMatrix, &ldc);
    }

    // Convert dot products to Euclidean distances: dist[i,j] = sqrt(max(0, norm[i]+norm[j]-2*dot))
    const size_t distBlockSize = 256;
    const size_t nDistBlocks   = nRows / distBlockSize + (nRows % distBlockSize > 0);

    daal::threader_for(nDistBlocks, nDistBlocks, [&](size_t iBlock) {
        const size_t i_begin = iBlock * distBlockSize;
        const size_t i_end   = (i_begin + distBlockSize > nRows) ? nRows : i_begin + distBlockSize;

        for (size_t i = i_begin; i < i_end; i++)
        {
            algorithmFPType * row    = distMatrix + i * nRows;
            const algorithmFPType ni = norms[i];
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nRows; j++)
            {
                algorithmFPType d2 = ni + norms[j] - algorithmFPType(2) * row[j];
                if (d2 < algorithmFPType(0)) d2 = algorithmFPType(0);
                row[j] = static_cast<algorithmFPType>(sqrt(static_cast<double>(d2)));
            }
            row[i] = algorithmFPType(0);
        }
    });

    // =========================================================================
    // Step 2: Compute core distances (parallelized with threader_for)
    // =========================================================================

    daal::services::internal::TArray<algorithmFPType, cpu> coreDists(nRows);
    algorithmFPType * coreDistances = coreDists.get();
    DAAL_CHECK_MALLOC(coreDistances);

    const size_t target = (minSamples > 0) ? minSamples - 1 : 0;

    daal::threader_for(nRows, nRows, [&](size_t i) {
        // Copy row from distance matrix
        std::vector<algorithmFPType> dists(nRows);
        const algorithmFPType * row = distMatrix + i * nRows;
        for (size_t j = 0; j < nRows; j++) dists[j] = row[j];

        size_t t = (target >= nRows) ? nRows - 1 : target;
        std::nth_element(dists.begin(), dists.begin() + t, dists.end());
        coreDistances[i] = dists[t];
    });

    // =========================================================================
    // Step 3: Build MST using Prim's algorithm with MRD
    // =========================================================================

    daal::services::internal::TArray<int, cpu> mstFromArr(edgeCount);
    daal::services::internal::TArray<int, cpu> mstToArr(edgeCount);
    daal::services::internal::TArray<algorithmFPType, cpu> mstWeightsArr(edgeCount);
    int * mstFrom                = mstFromArr.get();
    int * mstTo                  = mstToArr.get();
    algorithmFPType * mstWeights = mstWeightsArr.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    {
        std::vector<algorithmFPType> minEdge(nRows, std::numeric_limits<algorithmFPType>::max());
        std::vector<int> minFrom(nRows, 0);
        std::vector<bool> inMst(nRows, false);

        inMst[0]                     = true;
        const algorithmFPType * row0 = distMatrix;
        const algorithmFPType core0  = coreDistances[0];
        for (size_t j = 1; j < nRows; j++)
        {
            algorithmFPType mrd = row0[j];
            if (core0 > mrd) mrd = core0;
            if (coreDistances[j] > mrd) mrd = coreDistances[j];
            minEdge[j] = mrd;
        }

        for (size_t e = 0; e < edgeCount; e++)
        {
            int best              = -1;
            algorithmFPType bestW = std::numeric_limits<algorithmFPType>::max();
            for (size_t j = 0; j < nRows; j++)
            {
                if (!inMst[j] && minEdge[j] < bestW)
                {
                    bestW = minEdge[j];
                    best  = static_cast<int>(j);
                }
            }

            mstFrom[e]    = minFrom[best];
            mstTo[e]      = best;
            mstWeights[e] = bestW;
            inMst[best]   = true;

            const algorithmFPType * bestRow = distMatrix + static_cast<size_t>(best) * nRows;
            const algorithmFPType coreBest  = coreDistances[best];
            for (size_t j = 0; j < nRows; j++)
            {
                if (inMst[j]) continue;
                algorithmFPType mrd = bestRow[j];
                if (coreBest > mrd) mrd = coreBest;
                if (coreDistances[j] > mrd) mrd = coreDistances[j];
                if (mrd < minEdge[j])
                {
                    minEdge[j] = mrd;
                    minFrom[j] = best;
                }
            }
        }
    }

    // =========================================================================
    // Step 4: Sort MST edges by weight
    // =========================================================================

    {
        std::vector<size_t> order(edgeCount);
        std::iota(order.begin(), order.end(), size_t(0));
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return mstWeights[a] < mstWeights[b]; });

        std::vector<int> sortedFrom(edgeCount);
        std::vector<int> sortedTo(edgeCount);
        std::vector<algorithmFPType> sortedWeights(edgeCount);
        for (size_t i = 0; i < edgeCount; i++)
        {
            sortedFrom[i]    = mstFrom[order[i]];
            sortedTo[i]      = mstTo[order[i]];
            sortedWeights[i] = mstWeights[order[i]];
        }
        for (size_t i = 0; i < edgeCount; i++)
        {
            mstFrom[i]    = sortedFrom[i];
            mstTo[i]      = sortedTo[i];
            mstWeights[i] = sortedWeights[i];
        }
    }

    // =========================================================================
    // Step 5: Extract clusters (dendrogram -> condensed tree -> EOM -> labels)
    // =========================================================================

    const size_t nDendroNodes = nRows - 1;
    const size_t totalNodes   = 2 * nRows - 1;

    // Phase 1: Build Kruskal dendrogram via union-find
    std::vector<int> ufParent(nRows);
    std::vector<int> compSize(nRows, 1);
    std::iota(ufParent.begin(), ufParent.end(), 0);

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
    std::vector<DendroNode> dendro(nDendroNodes);
    std::vector<int> compToNode(nRows);
    std::iota(compToNode.begin(), compToNode.end(), 0);

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
    std::vector<int> nodeSize(totalNodes, 0);
    for (size_t i = 0; i < nRows; i++) nodeSize[i] = 1;
    for (size_t e = 0; e < nDendroNodes; e++) nodeSize[nRows + e] = dendro[e].size;

    std::vector<int> leftChild(totalNodes, -1);
    std::vector<int> rightChild(totalNodes, -1);
    std::vector<algorithmFPType> nodeWeight(totalNodes, algorithmFPType(0));
    for (size_t e = 0; e < nDendroNodes; e++)
    {
        const size_t nid = nRows + e;
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
    std::vector<CondensedEdge> condensed;
    condensed.reserve(2 * nRows);

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
        WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(assignBlock);
        int * assignments = assignBlock.get();
        for (size_t i = 0; i < nRows; i++) assignments[i] = -1;
        WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(ncBlock);
        ncBlock.get()[0] = 0;
        return services::Status();
    }

    int nextCid = static_cast<int>(nRows);
    std::vector<int> dendroToCluster(totalNodes, -1);
    dendroToCluster[root] = nextCid++;

    // Iterative leaf collection using explicit stack
    auto collectLeaves = [&](int startNid, std::vector<int> & out) {
        std::vector<int> traverseStack;
        traverseStack.push_back(startNid);
        while (!traverseStack.empty())
        {
            int nid = traverseStack.back();
            traverseStack.pop_back();
            if (nid < static_cast<int>(nRows))
            {
                out.push_back(nid);
            }
            else
            {
                if (rightChild[nid] >= 0) traverseStack.push_back(rightChild[nid]);
                if (leftChild[nid] >= 0) traverseStack.push_back(leftChild[nid]);
            }
        }
    };

    struct StackItem
    {
        int node;
        int cluster;
    };
    std::vector<StackItem> stack;
    stack.push_back({ root, dendroToCluster[root] });

    const int mcs = static_cast<int>(minClusterSize);

    while (!stack.empty())
    {
        auto item = stack.back();
        stack.pop_back();
        int nid       = item.node;
        int parentCid = item.cluster;

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
            const int lcid      = nextCid++;
            const int rcid      = nextCid++;
            dendroToCluster[lc] = lcid;
            dendroToCluster[rc] = rcid;
            condensed.push_back({ parentCid, lcid, lambda, ls });
            condensed.push_back({ parentCid, rcid, lambda, rs });
            stack.push_back({ lc, lcid });
            stack.push_back({ rc, rcid });
        }
        else if (lBig)
        {
            dendroToCluster[lc] = parentCid;
            std::vector<int> fallen;
            collectLeaves(rc, fallen);
            for (auto pt : fallen) condensed.push_back({ parentCid, pt, lambda, 1 });
            stack.push_back({ lc, parentCid });
        }
        else if (rBig)
        {
            dendroToCluster[rc] = parentCid;
            std::vector<int> fallen;
            collectLeaves(lc, fallen);
            for (auto pt : fallen) condensed.push_back({ parentCid, pt, lambda, 1 });
            stack.push_back({ rc, parentCid });
        }
        else
        {
            std::vector<int> fallen;
            collectLeaves(lc, fallen);
            collectLeaves(rc, fallen);
            for (auto pt : fallen) condensed.push_back({ parentCid, pt, lambda, 1 });
        }
    }

    // Phase 3: Stability and EOM selection
    const int nClusters = nextCid;
    const int rootCid   = static_cast<int>(nRows);

    std::vector<algorithmFPType> stability(nClusters, algorithmFPType(0));
    std::vector<algorithmFPType> lambdaBirth(nClusters, algorithmFPType(0));
    std::vector<bool> isLeafCluster(nClusters, true);
    std::vector<std::vector<int> > childClusters(nClusters);
    std::vector<int> clusterSize(nClusters, 0);
    clusterSize[rootCid] = static_cast<int>(nRows);

    for (const auto & e : condensed)
    {
        if (e.child >= static_cast<int>(nRows))
        {
            lambdaBirth[e.child]    = e.lambdaVal;
            isLeafCluster[e.parent] = false;
            childClusters[e.parent].push_back(e.child);
            clusterSize[e.child] = e.childSize;
        }
    }

    for (const auto & e : condensed)
    {
        const algorithmFPType birth   = lambdaBirth[e.parent];
        const algorithmFPType contrib = (e.lambdaVal - birth) * static_cast<algorithmFPType>(e.childSize);
        if (contrib > algorithmFPType(0)) stability[e.parent] += contrib;
    }

    std::vector<bool> isSelected(nClusters, true);
    for (int c = rootCid; c < nClusters; c++)
    {
        if (clusterSize[c] < mcs) isSelected[c] = false;
    }

    for (int c = nClusters - 1; c >= rootCid; c--)
    {
        if (isLeafCluster[c]) continue;

        algorithmFPType childSum = algorithmFPType(0);
        for (auto cc : childClusters[c]) childSum += stability[cc];

        if (childSum > stability[c])
        {
            isSelected[c] = false;
            stability[c]  = childSum;
        }
        else
        {
            std::vector<int> descStack;
            for (auto cc : childClusters[c]) descStack.push_back(cc);
            while (!descStack.empty())
            {
                auto d = descStack.back();
                descStack.pop_back();
                isSelected[d] = false;
                for (auto dd : childClusters[d]) descStack.push_back(dd);
            }
        }
    }

    // Phase 4: Label points
    int labelCounter = 0;
    std::vector<int> clusterLabel(nClusters, -1);
    for (int c = rootCid; c < nClusters; c++)
    {
        if (isSelected[c]) clusterLabel[c] = labelCounter++;
    }

    std::vector<int> clusterParent(nClusters, -1);
    for (const auto & e : condensed)
    {
        if (e.child >= static_cast<int>(nRows)) clusterParent[e.child] = e.parent;
    }

    std::vector<int> pointFellFrom(nRows, -1);
    for (const auto & e : condensed)
    {
        if (e.child < static_cast<int>(nRows)) pointFellFrom[e.child] = e.parent;
    }

    // Write assignments
    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    for (size_t i = 0; i < nRows; i++)
    {
        assignments[i] = -1;
        int c          = pointFellFrom[i];
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

    // Write number of clusters
    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
