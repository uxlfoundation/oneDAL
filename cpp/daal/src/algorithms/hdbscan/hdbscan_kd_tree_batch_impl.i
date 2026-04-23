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
// Distance functors for parameterizing kd-tree queries by metric
// Each provides:
//   pointDist(a, b, nCols)       — full point-to-point distance
//   bboxLowerBound(q, lo, hi, nCols) — minimum distance from query to bbox
//   planeDist(diff)              — distance to splitting hyperplane
// =========================================================================
template <typename FPType>
struct EuclideanDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            const FPType diff = a[d] - b[d];
            sum += diff * diff;
        }
        return static_cast<FPType>(sqrt(static_cast<double>(sum)));
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            sum += excess * excess;
        }
        return static_cast<FPType>(sqrt(static_cast<double>(sum)));
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct ManhattanDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType diff = a[d] - b[d];
            sum += (diff >= FPType(0)) ? diff : -diff;
        }
        return sum;
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            sum += excess;
        }
        return sum;
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct MinkowskiDist
{
    double p;
    double invp;
    MinkowskiDist(double degree) : p(degree), invp(1.0 / degree) {}

    FPType pointDist(const FPType * a, const FPType * b, int nCols) const
    {
        double sum = 0.0;
        for (int d = 0; d < nCols; d++)
        {
            double diff = static_cast<double>(a[d] - b[d]);
            if (diff < 0.0) diff = -diff;
            sum += pow(diff, p);
        }
        return static_cast<FPType>(pow(sum, invp));
    }
    FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols) const
    {
        double sum = 0.0;
        for (int d = 0; d < nCols; d++)
        {
            double excess = 0.0;
            if (query[d] < lo[d])
                excess = static_cast<double>(lo[d] - query[d]);
            else if (query[d] > hi[d])
                excess = static_cast<double>(query[d] - hi[d]);
            sum += pow(excess, p);
        }
        return static_cast<FPType>(pow(sum, invp));
    }
    FPType planeDist(FPType diff) const { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct ChebyshevDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType mx = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType diff = a[d] - b[d];
            if (diff < FPType(0)) diff = -diff;
            if (diff > mx) mx = diff;
        }
        return mx;
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType mx = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            if (excess > mx) mx = excess;
        }
        return mx;
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

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
// k-NN query on kd-tree (templated on distance functor)
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static void knnQuery(const algorithmFPType * data, int nCols, const KdNode<algorithmFPType> * nodes, const int * pointIndices,
                     const algorithmFPType * queryPoint, int nodeIdx, KnnHeap<algorithmFPType> & heap, const DistFunc & distFunc)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    if (node.splitDim < 0)
    {
        // Leaf node: check all points
        for (int i = node.pointBegin; i < node.pointEnd; i++)
        {
            const int pi                = pointIndices[i];
            const algorithmFPType * row = data + pi * nCols;
            const algorithmFPType dist  = distFunc.pointDist(queryPoint, row, nCols);
            heap.push(dist, pi);
        }
        return;
    }

    const algorithmFPType queryVal = queryPoint[node.splitDim];
    const algorithmFPType diff     = queryVal - node.splitVal;

    const int nearChild = (diff <= 0) ? node.left : node.right;
    const int farChild  = (diff <= 0) ? node.right : node.left;

    knnQuery(data, nCols, nodes, pointIndices, queryPoint, nearChild, heap, distFunc);

    // Prune: only visit far subtree if the splitting plane is closer than current k-th NN
    const algorithmFPType pd = distFunc.planeDist(diff);
    if (pd < heap.maxDist())
    {
        knnQuery(data, nCols, nodes, pointIndices, queryPoint, farChild, heap, distFunc);
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
// Templated on distance functor to support multiple metrics.
// =========================================================================
template <typename algorithmFPType, typename DistFunc>
static void nearestMrdBoruvkaQuery(const algorithmFPType * data, int nCols, const KdNode<algorithmFPType> * nodes, const int * pointIndices,
                                   const algorithmFPType * coreDistances, const algorithmFPType * bboxLo, const algorithmFPType * bboxHi,
                                   const algorithmFPType * minCoreDistNode, const int * componentOf, const algorithmFPType * queryPoint, int queryIdx,
                                   algorithmFPType queryCoreD, int queryComponent, int nodeIdx, algorithmFPType & bestMrd, int & bestIdx,
                                   const DistFunc & distFunc)
{
    const KdNode<algorithmFPType> & node = nodes[nodeIdx];

    // Pruning 1: all points in this subtree belong to the same component as query
    if (node.componentId == queryComponent) return;

    // Pruning 2: bounding-box MRD lower bound
    {
        const algorithmFPType * lo    = bboxLo + nodeIdx * nCols;
        const algorithmFPType * hi    = bboxHi + nodeIdx * nCols;
        const algorithmFPType minDist = distFunc.bboxLowerBound(queryPoint, lo, hi, nCols);

        // MRD(q, p) = max(core_q, core_p, dist(q,p)) >= max(core_q, minCoreDist_subtree, minDist)
        algorithmFPType mrdLB = minDist;
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
            const algorithmFPType dist  = distFunc.pointDist(queryPoint, row, nCols);

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
                           queryCoreD, queryComponent, nearChild, bestMrd, bestIdx, distFunc);
    nearestMrdBoruvkaQuery(data, nCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf, queryPoint, queryIdx,
                           queryCoreD, queryComponent, farChild, bestMrd, bestIdx, distFunc);
}

// =========================================================================
// Core distances + Boruvka MST, templated on distance functor
// =========================================================================
template <typename algorithmFPType, CpuType cpu, typename DistFunc>
static void computeCoreDistAndMst(const algorithmFPType * data, size_t nRows, size_t nCols, size_t minSamples, KdNode<algorithmFPType> * nodes,
                                  int * pointIndices, int totalTreeNodes, algorithmFPType * bboxLo, algorithmFPType * bboxHi,
                                  algorithmFPType * coreDistances, int * mstFrom, int * mstTo, algorithmFPType * mstWeights,
                                  const DistFunc & distFunc)
{
    const int k = static_cast<int>(minSamples);

    // Step 2: Compute core distances via k-NN queries on the kd-tree
    {
        daal::TlsMem<algorithmFPType, cpu> tlsHeapDists(k);
        daal::TlsMem<int, cpu> tlsHeapIndices(k);

        daal::threader_for(static_cast<int>(nRows), static_cast<int>(nRows), [&](size_t i) {
            algorithmFPType * heapDists = tlsHeapDists.local();
            int * heapIndices           = tlsHeapIndices.local();
            if (!heapDists || !heapIndices) return;

            KnnHeap<algorithmFPType> heap;
            heap.init(heapDists, heapIndices, k);

            knnQuery(data, static_cast<int>(nCols), nodes, pointIndices, data + i * nCols, 0, heap, distFunc);

            coreDistances[i] = heap.maxDist();
        });
    }

    // Step 2b: Per-node minimum core distances
    daal::services::internal::TArray<algorithmFPType, cpu> minCoreDistNodeArr(totalTreeNodes);
    algorithmFPType * minCoreDistNode = minCoreDistNodeArr.get();
    if (!minCoreDistNode) return;
    computeMinCoreDists(nodes, pointIndices, coreDistances, minCoreDistNode, 0);

    // Step 3: Boruvka MST
    const size_t edgeCount = nRows - 1;

    daal::services::internal::TArray<int, cpu> ufParentArr(nRows);
    daal::services::internal::TArray<int, cpu> ufRankArr(nRows);
    daal::services::internal::TArray<int, cpu> componentOfArr(nRows);
    int * ufParent    = ufParentArr.get();
    int * ufRank      = ufRankArr.get();
    int * componentOf = componentOfArr.get();
    if (!ufParent || !ufRank || !componentOf) return;

    daal::services::internal::TArray<algorithmFPType, cpu> pointBestMrdArr(nRows);
    daal::services::internal::TArray<int, cpu> pointBestIdxArr(nRows);
    algorithmFPType * pointBestMrd = pointBestMrdArr.get();
    int * pointBestIdx             = pointBestIdxArr.get();
    if (!pointBestMrd || !pointBestIdx) return;

    daal::services::internal::TArray<algorithmFPType, cpu> compBestMrdArr(nRows);
    daal::services::internal::TArray<int, cpu> compBestFromArr(nRows);
    daal::services::internal::TArray<int, cpu> compBestToArr(nRows);
    algorithmFPType * compBestMrd = compBestMrdArr.get();
    int * compBestFrom            = compBestFromArr.get();
    int * compBestTo              = compBestToArr.get();
    if (!compBestMrd || !compBestFrom || !compBestTo) return;

    for (size_t i = 0; i < nRows; i++)
    {
        ufParent[i]    = static_cast<int>(i);
        ufRank[i]      = 0;
        componentOf[i] = static_cast<int>(i);
    }

    auto ufFind = [&](int x) -> int {
        while (ufParent[x] != x)
        {
            ufParent[x] = ufParent[ufParent[x]];
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

    updateNodeComponents(nodes, pointIndices, componentOf, 0);

    size_t edgesAdded    = 0;
    size_t numComponents = nRows;
    const int iNRows     = static_cast<int>(nRows);
    const int iNCols     = static_cast<int>(nCols);

    while (numComponents > 1)
    {
        daal::threader_for(iNRows, iNRows, [&](size_t i) {
            const int comp                   = componentOf[i];
            const algorithmFPType * queryPt  = data + i * nCols;
            const algorithmFPType queryCoreD = coreDistances[i];
            algorithmFPType bestMrd          = std::numeric_limits<algorithmFPType>::max();
            int bestIdx                      = -1;

            nearestMrdBoruvkaQuery(data, iNCols, nodes, pointIndices, coreDistances, bboxLo, bboxHi, minCoreDistNode, componentOf, queryPt,
                                   static_cast<int>(i), queryCoreD, comp, 0, bestMrd, bestIdx, distFunc);

            pointBestMrd[i] = bestMrd;
            pointBestIdx[i] = bestIdx;
        });

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

        size_t addedThisRound = 0;
        for (size_t c = 0; c < nRows; c++)
        {
            if (compBestFrom[c] < 0) continue;
            const int u  = compBestFrom[c];
            const int v  = compBestTo[c];
            const int ru = ufFind(u);
            const int rv = ufFind(v);
            if (ru == rv) continue;

            mstFrom[edgesAdded]    = u;
            mstTo[edgesAdded]      = v;
            mstWeights[edgesAdded] = compBestMrd[c];
            edgesAdded++;
            addedThisRound++;

            ufUnion(ru, rv);
            numComponents--;
        }

        if (addedThisRound == 0) break;

        for (size_t i = 0; i < nRows; i++)
        {
            componentOf[i] = ufFind(static_cast<int>(i));
        }

        updateNodeComponents(nodes, pointIndices, componentOf, 0);
    }
}

// =========================================================================
// Main HDBSCAN kd-tree compute
// =========================================================================
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           int metric, double degree)
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
    // Steps 2-3: Core distances + Boruvka MST (dispatched by metric)
    // =========================================================================

    daal::services::internal::TArray<algorithmFPType, cpu> coreDists(nRows);
    algorithmFPType * coreDistances = coreDists.get();
    DAAL_CHECK_MALLOC(coreDistances);

    daal::services::internal::TArray<int, cpu> mstFromArr(edgeCount);
    daal::services::internal::TArray<int, cpu> mstToArr(edgeCount);
    daal::services::internal::TArray<algorithmFPType, cpu> mstWeightsArr(edgeCount);
    int * mstFrom                = mstFromArr.get();
    int * mstTo                  = mstToArr.get();
    algorithmFPType * mstWeights = mstWeightsArr.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    switch (metric)
    {
    case manhattan:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, ManhattanDist<algorithmFPType> {});
        break;
    case minkowski:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, MinkowskiDist<algorithmFPType>(degree));
        break;
    case chebyshev:
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, ChebyshevDist<algorithmFPType> {});
        break;
    default: // euclidean
        computeCoreDistAndMst<algorithmFPType, cpu>(data, nRows, nCols, minSamples, nodes, pointIndices, totalTreeNodes, bboxLo, bboxHi,
                                                    coreDistances, mstFrom, mstTo, mstWeights, EuclideanDist<algorithmFPType> {});
        break;
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
