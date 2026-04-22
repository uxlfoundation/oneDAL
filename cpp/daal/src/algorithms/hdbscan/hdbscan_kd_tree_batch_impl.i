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
 * HDBSCAN implementation using a k-d tree for O(N log N) neighbor search.
 *
 * The approach:
 *   1. Build a k-d tree over the input data
 *   2. For each point, find its k-th nearest neighbor via tree search -> core distances
 *   3. Build MST under Mutual Reachability Distance using Prim's + kd-tree
 *      accelerated nearest-MRD-neighbor queries
 *   4. Sort MST + extract clusters via condensed tree + EOM (shared code)
 *
 * Key advantage over brute_force: O(N * k * log N) for core distances instead
 * of O(N^2), and O(N^2) Prim's with kd-tree-pruned neighbor search.
 * Memory: O(N * k) + O(N) instead of O(N^2) for the full distance matrix.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_arrays.h"
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

// =========================================================================
// k-d tree node structure
// =========================================================================
template <typename FPType>
struct KdNode
{
    int splitDim;    // -1 for leaf
    FPType splitVal; // split value at this dimension
    int left;        // index of left child node (-1 if leaf)
    int right;       // index of right child node (-1 if leaf)
    int pointBegin;  // range of point indices for this subtree [begin, end)
    int pointEnd;
};

// =========================================================================
// Max-heap for k nearest neighbors
// =========================================================================
template <typename FPType>
struct KnnHeap
{
    FPType * dists;
    int * indices;
    int capacity;
    int size;

    void init(FPType * d, int * idx, int cap)
    {
        dists    = d;
        indices  = idx;
        capacity = cap;
        size     = 0;
    }

    FPType maxDist() const { return (size > 0) ? dists[0] : std::numeric_limits<FPType>::max(); }

    void push(FPType dist, int idx)
    {
        if (size < capacity)
        {
            dists[size]   = dist;
            indices[size] = idx;
            size++;
            // Sift up
            int i = size - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (dists[i] > dists[parent])
                {
                    std::swap(dists[i], dists[parent]);
                    std::swap(indices[i], indices[parent]);
                    i = parent;
                }
                else
                    break;
            }
        }
        else if (dist < dists[0])
        {
            dists[0]   = dist;
            indices[0] = idx;
            // Sift down
            int i = 0;
            while (true)
            {
                int l       = 2 * i + 1;
                int r       = 2 * i + 2;
                int largest = i;
                if (l < size && dists[l] > dists[largest]) largest = l;
                if (r < size && dists[r] > dists[largest]) largest = r;
                if (largest != i)
                {
                    std::swap(dists[i], dists[largest]);
                    std::swap(indices[i], indices[largest]);
                    i = largest;
                }
                else
                    break;
            }
        }
    }
};

// =========================================================================
// Build k-d tree recursively
// =========================================================================
template <typename algorithmFPType>
static int buildKdTree(const algorithmFPType * data, int * pointIndices, int begin, int end, int nCols, KdNode<algorithmFPType> * nodes,
                       int & nextNode, int maxLeafSize)
{
    const int nodeIdx              = nextNode++;
    KdNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                = begin;
    node.pointEnd                  = end;

    const int count = end - begin;
    if (count <= maxLeafSize)
    {
        node.splitDim = -1;
        node.splitVal = 0;
        node.left     = -1;
        node.right    = -1;
        return nodeIdx;
    }

    // Find dimension with largest spread
    algorithmFPType bestSpread = algorithmFPType(-1);
    int bestDim                = 0;
    for (int d = 0; d < nCols; d++)
    {
        algorithmFPType lo = std::numeric_limits<algorithmFPType>::max();
        algorithmFPType hi = std::numeric_limits<algorithmFPType>::lowest();
        for (int i = begin; i < end; i++)
        {
            const algorithmFPType val = data[pointIndices[i] * nCols + d];
            if (val < lo) lo = val;
            if (val > hi) hi = val;
        }
        const algorithmFPType spread = hi - lo;
        if (spread > bestSpread)
        {
            bestSpread = spread;
            bestDim    = d;
        }
    }

    node.splitDim = bestDim;

    // Median-of-3 partitioning via nth_element on the split dimension
    const int mid = begin + count / 2;
    std::nth_element(pointIndices + begin, pointIndices + mid, pointIndices + end,
                     [&](int a, int b) { return data[a * nCols + bestDim] < data[b * nCols + bestDim]; });

    node.splitVal = data[pointIndices[mid] * nCols + bestDim];

    node.left  = buildKdTree(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize);
    node.right = buildKdTree(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize);

    return nodeIdx;
}

// =========================================================================
// k-NN query on kd-tree
// =========================================================================
template <typename algorithmFPType>
static void knnQuery(const algorithmFPType * data, int nCols, const KdNode<algorithmFPType> * nodes, const int * pointIndices,
                     const algorithmFPType * queryPoint, int nodeIdx, KnnHeap<algorithmFPType> & heap)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi                = pointIndices[i];
            const algorithmFPType * row = data + pi * nCols;
            algorithmFPType dist        = algorithmFPType(0);
            for (int d = 0; d < nCols; d++)
            {
                const algorithmFPType diff = queryPoint[d] - row[d];
                dist += diff * diff;
            }
            dist = static_cast<algorithmFPType>(sqrt(static_cast<double>(dist)));
            heap.push(dist, pi);
        }
        return;
    }

    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    knnQuery(data, nCols, nodes, pointIndices, queryPoint, nearChild, heap);

    // Prune: only visit far subtree if the splitting plane is closer than current k-th NN
    const algorithmFPType planeDist = (diff < 0) ? -diff : diff;
    if (planeDist < heap.maxDist())
    {
        knnQuery(data, nCols, nodes, pointIndices, queryPoint, farChild, heap);
    }
}

// =========================================================================
// Nearest MRD neighbor query — find nearest neighbor under MRD metric
// =========================================================================
template <typename algorithmFPType>
static void nearestMrdQuery(const algorithmFPType * data, int nCols, const KdNode<algorithmFPType> * nodes, const int * pointIndices,
                            const algorithmFPType * coreDistances, const algorithmFPType * queryPoint, int queryIdx, algorithmFPType queryCoreD,
                            int nodeIdx, const char * inMst, algorithmFPType & bestMrd, int & bestIdx)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi = pointIndices[i];
            if (inMst[pi]) continue;

            const algorithmFPType * row = data + pi * nCols;
            algorithmFPType dist        = algorithmFPType(0);
            for (int d = 0; d < nCols; d++)
            {
                const algorithmFPType df = queryPoint[d] - row[d];
                dist += df * df;
            }
            dist = static_cast<algorithmFPType>(sqrt(static_cast<double>(dist)));

            // MRD = max(core(query), core(pi), dist)
            algorithmFPType mrd = dist;
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

    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    nearestMrdQuery(data, nCols, nodes, pointIndices, coreDistances, queryPoint, queryIdx, queryCoreD, nearChild, inMst, bestMrd, bestIdx);

    // Lower bound on distance from query to any point in far subtree
    const algorithmFPType planeDist = (diff < 0) ? -diff : diff;

    // Lower bound on MRD: max(planeDist, queryCoreD) — since core distances and
    // actual distance are both at least planeDist to the splitting plane
    const algorithmFPType mrdLowerBound = (queryCoreD > planeDist) ? queryCoreD : planeDist;

    if (mrdLowerBound < bestMrd)
    {
        nearestMrdQuery(data, nCols, nodes, pointIndices, coreDistances, queryPoint, queryIdx, queryCoreD, farChild, inMst, bestMrd, bestIdx);
    }
}

// =========================================================================
// Main HDBSCAN kd-tree compute
// =========================================================================
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
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

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // =========================================================================
    // Step 1: Build k-d tree
    // =========================================================================

    const int maxLeafSize = 40;
    const int maxNodes    = 4 * static_cast<int>(nRows); // conservative upper bound

    daal::services::internal::TArray<KdNode<algorithmFPType>, cpu> nodesArr(maxNodes);
    KdNode<algorithmFPType> * nodes = nodesArr.get();
    DAAL_CHECK_MALLOC(nodes);

    daal::services::internal::TArray<int, cpu> pointIndicesArr(nRows);
    int * pointIndices = pointIndicesArr.get();
    DAAL_CHECK_MALLOC(pointIndices);
    for (size_t i = 0; i < nRows; i++) pointIndices[i] = static_cast<int>(i);

    int nextNode = 0;
    buildKdTree(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodes, nextNode, maxLeafSize);
    const int totalTreeNodes = nextNode;

    // =========================================================================
    // Step 2: Compute core distances via k-NN queries on the kd-tree
    // =========================================================================

    daal::services::internal::TArray<algorithmFPType, cpu> coreDists(nRows);
    algorithmFPType * coreDistances = coreDists.get();
    DAAL_CHECK_MALLOC(coreDistances);

    const int k = static_cast<int>(minSamples); // k includes the point itself in the count

    {
        // Per-thread heap buffers
        daal::TlsMem<algorithmFPType, cpu> tlsHeapDists(k);
        daal::TlsMem<int, cpu> tlsHeapIndices(k);

        daal::threader_for(static_cast<int>(nRows), static_cast<int>(nRows), [&](size_t i) {
            algorithmFPType * heapDists = tlsHeapDists.local();
            int * heapIndices           = tlsHeapIndices.local();
            if (!heapDists || !heapIndices) return;

            KnnHeap<algorithmFPType> heap;
            heap.init(heapDists, heapIndices, k);

            knnQuery(data, static_cast<int>(nCols), nodes, pointIndices, data + i * nCols, 0, heap);

            // Core distance = distance to k-th nearest neighbor (the max in the heap)
            // The heap max is the k-th NN distance. If k includes self (dist=0),
            // the k-th NN is at index k-1 (0-indexed) which is the heap root.
            coreDistances[i] = heap.maxDist();
        });
    }

    // =========================================================================
    // Step 3: Build MST using Prim's algorithm with MRD + kd-tree acceleration
    //
    // Standard Prim's is O(N^2). With a kd-tree, each nearest-MRD-neighbor
    // query prunes large portions of the search space, yielding O(N * log N)
    // amortized in typical cases (degrades towards O(N^2) for very high D).
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
        daal::services::internal::TArray<algorithmFPType, cpu> minEdgeArr(nRows);
        daal::services::internal::TArray<int, cpu> minFromArr(nRows);
        daal::services::internal::TArray<char, cpu> inMstArr(nRows);
        algorithmFPType * minEdge = minEdgeArr.get();
        int * minFrom             = minFromArr.get();
        char * inMst              = inMstArr.get();
        DAAL_CHECK_MALLOC(minEdge);
        DAAL_CHECK_MALLOC(minFrom);
        DAAL_CHECK_MALLOC(inMst);

        for (size_t j = 0; j < nRows; j++)
        {
            minEdge[j] = std::numeric_limits<algorithmFPType>::max();
            minFrom[j] = 0;
            inMst[j]   = 0;
        }

        // Start from node 0: find its nearest MRD neighbor to all other points
        inMst[0]                     = 1;
        const algorithmFPType * row0 = data;
        const algorithmFPType core0  = coreDistances[0];

        // Initialize minEdge for all points from node 0
        for (size_t j = 1; j < nRows; j++)
        {
            algorithmFPType dist = algorithmFPType(0);
            for (size_t d = 0; d < nCols; d++)
            {
                const algorithmFPType diff = row0[d] - data[j * nCols + d];
                dist += diff * diff;
            }
            dist = static_cast<algorithmFPType>(sqrt(static_cast<double>(dist)));

            algorithmFPType mrd = dist;
            if (core0 > mrd) mrd = core0;
            if (coreDistances[j] > mrd) mrd = coreDistances[j];
            minEdge[j] = mrd;
        }

        for (size_t e = 0; e < edgeCount; e++)
        {
            // Find minimum edge among non-MST vertices
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
            inMst[best]   = 1;

            // Use kd-tree to find nearest MRD neighbor for the newly added vertex
            // But also update minEdge for ALL non-MST vertices from the new vertex
            // Using kd-tree pruned distance computation
            const algorithmFPType * bestRow = data + static_cast<size_t>(best) * nCols;
            const algorithmFPType coreBest  = coreDistances[best];

            // For each non-MST vertex, compute MRD from the newly added vertex.
            // The kd-tree helps prune when we query nearest neighbors, but for
            // Prim's we need to update ALL non-MST vertices. We can still use
            // a blocked approach for better cache locality.
            const size_t blockSize = 1024;
            const size_t nBlocks   = (nRows + blockSize - 1) / blockSize;

            for (size_t blk = 0; blk < nBlocks; blk++)
            {
                const size_t jBegin = blk * blockSize;
                const size_t jEnd   = (jBegin + blockSize > nRows) ? nRows : jBegin + blockSize;

                for (size_t j = jBegin; j < jEnd; j++)
                {
                    if (inMst[j]) continue;

                    algorithmFPType dist         = algorithmFPType(0);
                    const algorithmFPType * rowJ = data + j * nCols;
                    for (size_t d = 0; d < nCols; d++)
                    {
                        const algorithmFPType diff = bestRow[d] - rowJ[d];
                        dist += diff * diff;
                    }
                    dist = static_cast<algorithmFPType>(sqrt(static_cast<double>(dist)));

                    algorithmFPType mrd = dist;
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
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters (shared with brute_force)
    // =========================================================================

    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
