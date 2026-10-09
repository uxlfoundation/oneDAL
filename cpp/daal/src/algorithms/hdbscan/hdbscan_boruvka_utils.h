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

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
