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
 * HDBSCAN implementation using a k-d tree with Boruvka's MST algorithm.
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
    int componentId; // -1 = mixed components, >= 0 = all points in same component
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
// Build k-d tree recursively with bounding box computation
// =========================================================================
template <typename algorithmFPType>
static int buildKdTree(const algorithmFPType * data, int * pointIndices, int begin, int end, int nCols, KdNode<algorithmFPType> * nodes,
                       int & nextNode, int maxLeafSize, algorithmFPType * bboxLo, algorithmFPType * bboxHi)
{
    const int nodeIdx              = nextNode++;
    KdNode<algorithmFPType> & node = nodes[nodeIdx];
    node.pointBegin                = begin;
    node.pointEnd                  = end;
    node.componentId               = -1;

    // Compute bounding box and find dimension with largest spread
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
        bboxLo[nodeIdx * nCols + d]  = lo;
        bboxHi[nodeIdx * nCols + d]  = hi;
        const algorithmFPType spread = hi - lo;
        if (spread > bestSpread)
        {
            bestSpread = spread;
            bestDim    = d;
        }
    }

    const int count = end - begin;
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
    const int mid = begin + count / 2;
    std::nth_element(pointIndices + begin, pointIndices + mid, pointIndices + end,
                     [&](int a, int b) { return data[a * nCols + bestDim] < data[b * nCols + bestDim]; });

    node.splitVal = data[pointIndices[mid] * nCols + bestDim];

    node.left  = buildKdTree(data, pointIndices, begin, mid, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);
    node.right = buildKdTree(data, pointIndices, mid, end, nCols, nodes, nextNode, maxLeafSize, bboxLo, bboxHi);

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
// Compute minimum core distance per tree node (bottom-up, O(N) total)
// =========================================================================
template <typename algorithmFPType>
static algorithmFPType computeMinCoreDists(const KdNode<algorithmFPType> * nodes, const int * pointIndices, const algorithmFPType * coreDistances,
                                           algorithmFPType * minCoreDistNode, int nodeIdx)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.splitDim < 0)
    {
        algorithmFPType minCD = std::numeric_limits<algorithmFPType>::max();
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const algorithmFPType cd = coreDistances[pointIndices[i]];
            if (cd < minCD) minCD = cd;
        }
        minCoreDistNode[nodeIdx] = minCD;
        return minCD;
    }
    const algorithmFPType leftMin  = computeMinCoreDists(nodes, pointIndices, coreDistances, minCoreDistNode, node.left);
    const algorithmFPType rightMin = computeMinCoreDists(nodes, pointIndices, coreDistances, minCoreDistNode, node.right);
    minCoreDistNode[nodeIdx]       = (leftMin < rightMin) ? leftMin : rightMin;
    return minCoreDistNode[nodeIdx];
}

// =========================================================================
// Update component IDs on tree nodes (bottom-up, O(tree_nodes) total)
// Returns the component ID if all points in subtree are in same component,
// or -1 if mixed.
// =========================================================================
template <typename algorithmFPType>
static int updateNodeComponents(KdNode<algorithmFPType> * nodes, const int * pointIndices, const int * componentOf, int nodeIdx)
{
    KdNode<algorithmFPType> & node = nodes[nodeIdx];
    if (node.splitDim < 0)
    {
        // Leaf: check if all points share the same component
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
    const int leftComp  = updateNodeComponents(nodes, pointIndices, componentOf, node.left);
    const int rightComp = updateNodeComponents(nodes, pointIndices, componentOf, node.right);
    if (leftComp >= 0 && leftComp == rightComp)
    {
        node.componentId = leftComp;
        return leftComp;
    }
    node.componentId = -1;
    return -1;
}

// =========================================================================
// Boruvka nearest-different-component query under MRD metric
// Uses three-level pruning: component, bounding-box MRD lower bound, near/far
// =========================================================================
template <typename algorithmFPType>
static void nearestMrdBoruvkaQuery(const algorithmFPType * data, int nCols, const KdNode<algorithmFPType> * nodes, const int * pointIndices,
                                   const algorithmFPType * coreDistances, const algorithmFPType * bboxLo, const algorithmFPType * bboxHi,
                                   const algorithmFPType * minCoreDistNode, const int * componentOf, const algorithmFPType * queryPoint, int queryIdx,
                                   algorithmFPType queryCoreD, int queryComponent, int nodeIdx, algorithmFPType & bestMrd, int & bestIdx)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    // Pruning 1: all points in this subtree belong to the same component as query
    if (node.componentId == queryComponent) return;

    // Pruning 2: bounding-box MRD lower bound
    {
        algorithmFPType minSqDist  = algorithmFPType(0);
        const algorithmFPType * lo = bboxLo + nodeIdx * nCols;
        const algorithmFPType * hi = bboxHi + nodeIdx * nCols;
        for (int d = 0; d < nCols; d++)
        {
            algorithmFPType excess = algorithmFPType(0);
            if (queryPoint[d] < lo[d])
                excess = lo[d] - queryPoint[d];
            else if (queryPoint[d] > hi[d])
                excess = queryPoint[d] - hi[d];
            minSqDist += excess * excess;
        }
        const algorithmFPType minEuclDist = static_cast<algorithmFPType>(sqrt(static_cast<double>(minSqDist)));

        // MRD(q, p) = max(core_q, core_p, dist(q,p)) >= max(core_q, minCoreDist_subtree, minEuclDist)
        algorithmFPType mrdLB = minEuclDist;
        if (queryCoreD > mrdLB) mrdLB = queryCoreD;
        if (minCoreDistNode[nodeIdx] > mrdLB) mrdLB = minCoreDistNode[nodeIdx];

        if (mrdLB >= bestMrd) return;
    }

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi = pointIndices[i];
            if (componentOf[pi] == queryComponent) continue;

            const algorithmFPType * row = data + pi * nCols;
            algorithmFPType dist        = algorithmFPType(0);
            for (int d = 0; d < nCols; d++)
            {
                const algorithmFPType df = queryPoint[d] - row[d];
                dist += df * df;
            }
            dist = static_cast<algorithmFPType>(sqrt(static_cast<double>(dist)));

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

    // Visit nearer child first for tighter bestMrd
    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    nearestMrdBoruvkaQuery(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf, queryPoint, queryIdx,
                           queryCoreD, queryComponent, nearChild, bestMrd, bestIdx);
    nearestMrdBoruvkaQuery(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf, queryPoint, queryIdx,
                           queryCoreD, queryComponent, farChild, bestMrd, bestIdx);
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
    // Step 1: Build k-d tree with bounding boxes
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

    // Bounding boxes stored in SoA layout: bboxLo[nodeIdx * nCols + d]
    daal::services::internal::TArray<algorithmFPType, cpu> bboxLoArr(static_cast<size_t>(maxNodes) * nCols);
    daal::services::internal::TArray<algorithmFPType, cpu> bboxHiArr(static_cast<size_t>(maxNodes) * nCols);
    algorithmFPType * bboxLo = bboxLoArr.get();
    algorithmFPType * bboxHi = bboxHiArr.get();
    DAAL_CHECK_MALLOC(bboxLo);
    DAAL_CHECK_MALLOC(bboxHi);

    int nextNode = 0;
    buildKdTree(data, pointIndices, 0, static_cast<int>(nRows), static_cast<int>(nCols), nodes, nextNode, maxLeafSize, bboxLo, bboxHi);
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
    // Step 2b: Compute per-node minimum core distances (for MRD pruning)
    // =========================================================================

    daal::services::internal::TArray<algorithmFPType, cpu> minCoreDistNodeArr(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeArr.get();
    DAAL_CHECK_MALLOC(minCoreDistNode);
    computeMinCoreDists(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // =========================================================================
    // Step 3: Build MST using Boruvka's algorithm with kd-tree acceleration
    //
    // O(log N) rounds, each round: N kd-tree queries at O(log N) amortized.
    // Total: O(N * log^2 N), a major improvement over O(N^2) Prim's.
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
        // Union-find arrays
        daal::services::internal::TArray<int, cpu> ufParentArr(nRows);
        daal::services::internal::TArray<int, cpu> ufRankArr(nRows);
        daal::services::internal::TArray<int, cpu> componentOfArr(nRows);
        int * ufParent    = ufParentArr.get();
        int * ufRank      = ufRankArr.get();
        int * componentOf = componentOfArr.get();
        DAAL_CHECK_MALLOC(ufParent);
        DAAL_CHECK_MALLOC(ufRank);
        DAAL_CHECK_MALLOC(componentOf);

        // Per-point best neighbor (per round)
        daal::services::internal::TArray<algorithmFPType, cpu> pointBestMrdArr(nRows);
        daal::services::internal::TArray<int, cpu> pointBestIdxArr(nRows);
        algorithmFPType * pointBestMrd = pointBestMrdArr.get();
        int * pointBestIdx             = pointBestIdxArr.get();
        DAAL_CHECK_MALLOC(pointBestMrd);
        DAAL_CHECK_MALLOC(pointBestIdx);

        // Per-component best edge (per round)
        daal::services::internal::TArray<algorithmFPType, cpu> compBestMrdArr(nRows);
        daal::services::internal::TArray<int, cpu> compBestFromArr(nRows);
        daal::services::internal::TArray<int, cpu> compBestToArr(nRows);
        algorithmFPType * compBestMrd = compBestMrdArr.get();
        int * compBestFrom            = compBestFromArr.get();
        int * compBestTo              = compBestToArr.get();
        DAAL_CHECK_MALLOC(compBestMrd);
        DAAL_CHECK_MALLOC(compBestFrom);
        DAAL_CHECK_MALLOC(compBestTo);

        // Initialize: each point is its own component
        for (size_t i = 0; i < nRows; i++)
        {
            ufParent[i]    = static_cast<int>(i);
            ufRank[i]      = 0;
            componentOf[i] = static_cast<int>(i);
        }

        // Union-find with path compression + union by rank
        auto ufFind = [&](int x) -> int {
            while (ufParent[x] != x)
            {
                ufParent[x] = ufParent[ufParent[x]]; // path halving
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

        // Initial component IDs on tree nodes
        updateNodeComponents(nodes, pointIndices, componentOf, 0);

        size_t edgesAdded    = 0;
        size_t numComponents = nRows;
        const int iNRows     = static_cast<int>(nRows);
        const int iNCols     = static_cast<int>(nCols);

        while (numComponents > 1)
        {
            // Phase 1: For each point, find nearest different-component neighbor
            daal::threader_for(iNRows, iNRows, [&](size_t i) {
                const int comp                   = componentOf[i];
                const algorithmFPType * queryPt  = data + i * nCols;
                const algorithmFPType queryCoreD = coreDistances[i];
                algorithmFPType bestMrd          = std::numeric_limits<algorithmFPType>::max();
                int bestIdx                      = -1;

                nearestMrdBoruvkaQuery(data, iNCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf, queryPt,
                                       static_cast<int>(i), queryCoreD, comp, 0, bestMrd, bestIdx);

                pointBestMrd[i] = bestMrd;
                pointBestIdx[i] = bestIdx;
            });

            // Phase 2: For each component, find the lightest edge
            for (size_t i = 0; i < nRows; i++)
            {
                compBestMrd[i]  = std::numeric_limits<algorithmFPType>::max();
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

            // Phase 3: Add edges and merge components (deduplicate)
            size_t addedThisRound = 0;
            for (size_t c = 0; c < nRows; c++)
            {
                if (compBestFrom[c] < 0) continue;
                const int u  = compBestFrom[c];
                const int v  = compBestTo[c];
                const int ru = ufFind(u);
                const int rv = ufFind(v);
                if (ru == rv) continue; // already merged by a symmetric edge

                mstFrom[edgesAdded]    = u;
                mstTo[edgesAdded]      = v;
                mstWeights[edgesAdded] = compBestMrd[c];
                edgesAdded++;
                addedThisRound++;

                ufUnion(ru, rv);
                numComponents--;
            }

            if (addedThisRound == 0) break; // safety: no progress possible

            // Phase 4: Flatten component labels
            for (size_t i = 0; i < nRows; i++)
            {
                componentOf[i] = ufFind(static_cast<int>(i));
            }

            // Phase 5: Update tree node component IDs for pruning
            updateNodeComponents(nodes, pointIndices, componentOf, 0);
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
