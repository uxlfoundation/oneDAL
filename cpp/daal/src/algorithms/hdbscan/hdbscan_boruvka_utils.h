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

#include "services/daal_defines.h"
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

/// Union-Find (disjoint-set) with path halving + union-by-rank.
///
/// Shared across all three HDBSCAN CPU backends (brute-force / kd-tree /
/// ball-tree). Only phase 1 of Boruvka's MST (nearest-different-component MRD
/// search) legitimately differs across methods -- the union-find and the
/// per-round reduce / merge / component-id refresh sequence below is identical.
/// Holds bare pointers; caller owns the two `DAAL_INT[nRows]` backing arrays.
struct UnionFind
{
    DAAL_INT * parent; ///< `parent[i]` is the parent index; roots satisfy `parent[i] == i`
    DAAL_INT * rank;   ///< Rank per root; ties broken by union-by-rank

    /// Path-halving find.
    ///
    /// @param[in] x Element id
    ///
    /// @return Root id of the set containing `x`
    DAAL_INT find(DAAL_INT x) const
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

/// Reduce per-point candidate edges to per-component best edges.
///
/// Phase 2 of a Boruvka round: for each point `i`, `pointBestMrd[i]` /
/// `pointBestIdx[i]` are the best different-component candidate found in
/// phase 1. Reduces them into a per-component slot (indexed by
/// `componentOf[i]`), keeping the smallest MRD per component.
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

/// Emit the per-component best edges into the MST and merge components.
///
/// Phase 3 of a Boruvka round. Iterates every component slot: if it holds a
/// candidate whose endpoints resolve to two different UF roots, appends the
/// edge to the MST arrays, unions the roots, and decrements `numComponents`.
/// Ties (both endpoints already in the same component after another edge from
/// the same round unified them) are skipped.
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
        const DAAL_INT ru = uf.find(u);
        const DAAL_INT rv = uf.find(v);
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

/// Refresh per-point component ids after phase 3 unified some roots.
///
/// Phase 4 of a Boruvka round. Parallelized because the map is O(N) with
/// independent entries.
///
/// @tparam cpu CPU dispatch tag
///
/// @param[in]  nRows       Number of points
/// @param[in]  uf          Union-find state
/// @param[out] componentOf Per-point component id (written for every entry)
template <daal::internal::CpuType cpu>
static void refreshComponentIds(size_t nRows, const UnionFind & uf, DAAL_INT * componentOf)
{
    daal::threader_for(nRows, nRows, [&](size_t i) { componentOf[i] = uf.find(static_cast<DAAL_INT>(i)); });
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
