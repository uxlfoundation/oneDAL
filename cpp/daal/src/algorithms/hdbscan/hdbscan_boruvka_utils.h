/* file: hdbscan_boruvka_utils.h */
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

#pragma once

#include "services/daal_defines.h" // DAAL_MALLOC_DEFAULT_ALIGNMENT
#include "services/error_handling.h"
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

/// Union-find with path halving and union by rank over caller-owned arrays of length `nRows`.
struct UnionFind
{
    DAAL_INT * parent; ///< `parent[i]` is the parent index; roots satisfy `parent[i] == i`
    DAAL_INT * rank;   ///< Rank per root; ties broken by union-by-rank

    /// Find without path compression, safe to call concurrently on a shared instance.
    ///
    /// @param[in] x Element id
    ///
    /// @return Root id of the set containing `x`
    DAAL_INT find(DAAL_INT x) const
    {
        while (parent[x] != x)
        {
            x = parent[x];
        }
        return x;
    }

    /// Path-halving find; not safe to call concurrently on a shared instance.
    ///
    /// @param[in] x Element id
    ///
    /// @return Root id of the set containing `x`
    DAAL_INT findCompress(DAAL_INT x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x         = parent[x];
        }
        return x;
    }

    /// Union by rank; caller must pass root ids (i.e. `find()` outputs).
    ///
    /// @param[in] rx Root id 1
    /// @param[in] ry Root id 2
    void unionRoots(DAAL_INT rx, DAAL_INT ry)
    {
        if (rank[rx] < rank[ry])
            parent[rx] = ry;
        else if (rank[rx] > rank[ry])
            parent[ry] = rx;
        else
        {
            parent[ry] = rx;
            rank[rx]++;
        }
    }
};

/// Reduce per-point candidate edges to the smallest-MRD edge of each component.
///
/// @tparam FPType Floating-point type used for edge weights (MRD)
///
/// @param[in]  nRows        Number of points
/// @param[in]  componentOf  Per-point component id
/// @param[in]  pointBestMrd Per-point best MRD found in phase 1
/// @param[in]  pointBestIdx Per-point best target id from phase 1 (-1 if none)
/// @param[out] compBestMrd  Per-component best MRD, seeded internally to `+inf`
/// @param[out] compBestFrom Per-component source id, seeded to `-1`
/// @param[out] compBestTo   Per-component target id, seeded to `-1`
template <typename FPType>
static void reduceComponentBestEdges(size_t nRows, const DAAL_INT * componentOf, const FPType * pointBestMrd, const DAAL_INT * pointBestIdx,
                                     FPType * compBestMrd, DAAL_INT * compBestFrom, DAAL_INT * compBestTo)
{
    const FPType inf = daal::services::internal::MaxVal<FPType>::get();
    // The arrays come from TArray / TArrayScalable, aligned to DAAL_MALLOC_DEFAULT_ALIGNMENT.
    PRAGMA_OMP_SIMD_ARGS(aligned(compBestMrd, compBestFrom, compBestTo : DAAL_MALLOC_DEFAULT_ALIGNMENT))
    for (size_t i = 0; i < nRows; i++)
    {
        compBestMrd[i]  = inf;
        compBestFrom[i] = -1;
        compBestTo[i]   = -1;
    }
    for (size_t i = 0; i < nRows; i++)
    {
        if (pointBestIdx[i] < 0) continue;
        const DAAL_INT comp = componentOf[i];
        if (pointBestMrd[i] < compBestMrd[comp])
        {
            compBestMrd[comp]  = pointBestMrd[i];
            compBestFrom[comp] = static_cast<DAAL_INT>(i);
            compBestTo[comp]   = pointBestIdx[i];
        }
    }
}

/// Append each component's best edge to the MST and merge its endpoints' sets, skipping edges
/// whose endpoints an earlier edge of the same round already joined.
///
/// @tparam FPType Floating-point type used for edge weights (MRD)
///
/// @param[in]     nRows         Number of points
/// @param[in]     compBestMrd   Per-component best MRD
/// @param[in]     compBestFrom  Per-component source id
/// @param[in]     compBestTo    Per-component target id
/// @param[in,out] uf            Union-find state
/// @param[out]    mstFrom       MST source ids, appended to at `edgesAdded`
/// @param[out]    mstTo         MST target ids, appended to at `edgesAdded`
/// @param[out]    mstWeights    MST edge weights, appended to at `edgesAdded`
/// @param[in,out] edgesAdded    Running edge count (advanced in place)
/// @param[in,out] numComponents Remaining component count (decremented in place)
///
/// @return Number of edges added this round; caller uses `0` to break the outer loop
template <typename FPType>
static size_t mergeComponentsEmitEdges(size_t nRows, const FPType * compBestMrd, const DAAL_INT * compBestFrom, const DAAL_INT * compBestTo,
                                       UnionFind & uf, DAAL_INT * mstFrom, DAAL_INT * mstTo, FPType * mstWeights, size_t & edgesAdded,
                                       size_t & numComponents)
{
    size_t addedThisRound = 0;
    for (size_t c = 0; c < nRows; c++)
    {
        if (compBestFrom[c] < 0) continue;
        const DAAL_INT u  = compBestFrom[c];
        const DAAL_INT v  = compBestTo[c];
        const DAAL_INT ru = uf.findCompress(u);
        const DAAL_INT rv = uf.findCompress(v);
        if (ru == rv) continue;

        mstFrom[edgesAdded]    = u;
        mstTo[edgesAdded]      = v;
        mstWeights[edgesAdded] = compBestMrd[c];
        edgesAdded++;
        addedThisRound++;

        uf.unionRoots(ru, rv);
        numComponents--;
    }
    return addedThisRound;
}

/// Recompute each point's component id and flatten the forest so every point points at its root.
///
/// @tparam cpu CPU dispatch tag
///
/// @param[in]     nRows       Number of points
/// @param[in,out] uf          Union-find state; its forest is left fully flattened
/// @param[out]    componentOf Per-point component id (written for every entry)
template <daal::internal::CpuType cpu>
static void refreshComponentIds(size_t nRows, UnionFind & uf, DAAL_INT * componentOf)
{
    daal::threader_for(nRows, 1, [&](size_t i) { componentOf[i] = uf.find(static_cast<DAAL_INT>(i)); });
    DAAL_INT * const parent = uf.parent;
    daal::threader_for(nRows, 1, [&](size_t i) { parent[i] = componentOf[i]; });
}

/// Width from which the tree bounds barely prune, so `primMst` replaces the Boruvka rounds.
constexpr size_t primMinCols = 32;

/// Below this many row values left to scan, a Prim step runs serially: a parallel-for costs more than the scan.
constexpr size_t primSerialWork = 32768;

/// Build the MST under MRD with Prim's algorithm, computing the distances on the fly.
///
/// Needs O(nRows x nCols) extra memory, for a compact copy of the rows outside the tree. Each step
/// scans them in fixed blocks of rows, so the result does not depend on the thread count.
///
/// @tparam FPType   Floating-point type
/// @tparam cpu      CPU dispatch tag
/// @tparam DistFunc Metric functor exposing `pointDist`
///
/// @param[in]  data          Row-major input, `nRows x nCols`
/// @param[in]  nRows         Number of points
/// @param[in]  nCols         Number of features
/// @param[in]  coreDistances Per-point core distances, length `nRows`
/// @param[in]  distFunc      Metric functor instance (unscaled metric)
/// @param[in]  invAlpha      `1.0 / alpha`, applied only to the distance term of the MRD
/// @param[out] mstFrom       Source endpoint per MST edge, length `nRows - 1`
/// @param[out] mstTo         Target endpoint per MST edge, length `nRows - 1`
/// @param[out] mstWeights    Edge weights (MRD), length `nRows - 1`
/// @param[out] edgesAdded    Number of edges emitted; fewer than `nRows - 1` only on non-finite input
///
/// @return `ErrorMemoryAllocationFailed` if a scratch allocation fails
template <typename FPType, daal::internal::CpuType cpu, typename DistFunc>
static services::Status primMst(const FPType * data, size_t nRows, size_t nCols, const FPType * coreDistances, const DistFunc & distFunc,
                                FPType invAlpha, DAAL_INT * mstFrom, DAAL_INT * mstTo, FPType * mstWeights, size_t & edgesAdded)
{
    using daal::services::internal::TArray;
    const FPType inf       = daal::services::internal::MaxVal<FPType>::get();
    const size_t blockSize = 64;
    const size_t outCount  = nRows - 1;
    const size_t maxBlocks = (outCount + blockSize - 1) / blockSize;
    edgesAdded             = 0;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, outCount, nCols * sizeof(FPType));

    // Row id, row, core distance and best edge of each point outside the tree. Removing a point
    // moves the last one into its slot.
    TArray<DAAL_INT, cpu> outIdVec(outCount);
    TArray<FPType, cpu> outRowsVec(outCount * nCols);
    TArray<FPType, cpu> outCoreVec(outCount);
    TArray<FPType, cpu> bestMrdVec(outCount);
    TArray<DAAL_INT, cpu> bestFromVec(outCount);
    TArray<FPType, cpu> blockMrdVec(maxBlocks);
    TArray<size_t, cpu> blockPosVec(maxBlocks);
    DAAL_INT * outId    = outIdVec.get();
    FPType * outRows    = outRowsVec.get();
    FPType * outCore    = outCoreVec.get();
    FPType * bestMrd    = bestMrdVec.get();
    DAAL_INT * bestFrom = bestFromVec.get();
    FPType * blockMrd   = blockMrdVec.get();
    size_t * blockPos   = blockPosVec.get();
    DAAL_CHECK_MALLOC(outId && outRows && outCore && bestMrd && bestFrom && blockMrd && blockPos);

    for (size_t p = 0; p < outCount; p++)
    {
        outId[p]    = static_cast<DAAL_INT>(p + 1);
        outCore[p]  = coreDistances[p + 1];
        bestMrd[p]  = inf;
        bestFrom[p] = -1;
    }
    const size_t rowsBytes = outCount * nCols * sizeof(FPType);
    daal::services::internal::daal_memcpy_s(outRows, rowsBytes, data + nCols, rowsBytes);

    DAAL_INT current = 0;
    for (size_t remaining = outCount; remaining > 0; remaining--)
    {
        const FPType * currentRow = data + current * nCols;
        const FPType currentCore  = coreDistances[current];
        const size_t blockCount   = (remaining + blockSize - 1) / blockSize;

        const auto scanBlock = [&](size_t b) {
            const size_t begin = b * blockSize;
            const size_t end   = (begin + blockSize < remaining) ? begin + blockSize : remaining;
            FPType localMrd    = inf;
            size_t localPos    = remaining;
            for (size_t p = begin; p < end; p++)
            {
                // The MRD is at least both core distances, so only a pair below both needs the distance.
                const FPType coreP = outCore[p];
                if (currentCore < bestMrd[p] && coreP < bestMrd[p])
                {
                    FPType mrd = distFunc.template pointDist<cpu>(currentRow, outRows + p * nCols, nCols) * invAlpha;
                    mrd        = (currentCore > mrd) ? currentCore : mrd;
                    mrd        = (coreP > mrd) ? coreP : mrd;
                    if (mrd < bestMrd[p])
                    {
                        bestMrd[p]  = mrd;
                        bestFrom[p] = current;
                    }
                }
                if (bestMrd[p] < localMrd)
                {
                    localMrd = bestMrd[p];
                    localPos = p;
                }
            }
            blockMrd[b] = localMrd;
            blockPos[b] = localPos;
        };
        if (blockCount == 1 || remaining * nCols < primSerialWork)
        {
            for (size_t b = 0; b < blockCount; b++) scanBlock(b);
        }
        else
        {
            // A static partition keeps each thread on mostly the same rows from step to step.
            daal::static_threader_for(blockCount, [&](size_t b, size_t) { scanBlock(b); });
        }

        FPType minMrd = inf;
        size_t minPos = remaining;
        for (size_t b = 0; b < blockCount; b++)
        {
            if (blockMrd[b] < minMrd)
            {
                minMrd = blockMrd[b];
                minPos = blockPos[b];
            }
        }
        // Only non-finite input leaves every remaining point without an edge.
        if (minPos == remaining) break;

        current                = outId[minPos];
        mstFrom[edgesAdded]    = bestFrom[minPos];
        mstTo[edgesAdded]      = current;
        mstWeights[edgesAdded] = minMrd;
        edgesAdded++;

        const size_t last = remaining - 1;
        outId[minPos]     = outId[last];
        outCore[minPos]   = outCore[last];
        bestMrd[minPos]   = bestMrd[last];
        bestFrom[minPos]  = bestFrom[last];
        for (size_t d = 0; d < nCols; d++) outRows[minPos * nCols + d] = outRows[last * nCols + d];
    }
    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
