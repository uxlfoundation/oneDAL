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

/*
 * HDBSCAN brute-force implementation.
 *
 * The approach:
 *   1. Compute full pairwise distance matrix via GEMM
 *   2. Compute core distances (k-th nearest neighbor distance per point) with a
 *      bounded selection heap over each row -- no row copy, O(minSamples) scratch
 *   3. Build MST under Mutual Reachability Distance using Boruvka's algorithm,
 *      applying 1/alpha only to the dist term inside MRD = max(coreI, coreJ, dist/alpha)
 *   4. Sort MST + extract clusters via condensed tree + EOM/leaf (shared code)
 *
 * Complexity: O(N^2) for the distance matrix, O(N^2 * log N) worst case for
 *             Boruvka's MST.
 *
 * Boruvka is kept here even though Prim's would do strictly less total work on a
 * complete graph (one pass over the matrix instead of ~log2(N) passes), because
 * the two have very different parallel shapes and on this path the shape wins.
 * Boruvka's per-round nearest-different-component query is one
 * `threader_for(nRows)` over independent rows: N^2 work behind a single barrier,
 * spread over every thread. Prim's is N sequential barriers with only N
 * candidate slots to split between them, so it cannot fill a wide machine --
 * measured 20-25% slower than Boruvka across the brute-force benchmark set on a
 * 224-thread host at 15k-30k rows. See the note above Step 3.
 * Memory:     O(N^2) for the distance matrix, O(N) working arrays.
 */

#include <cstdint>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_boruvka_utils.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/hdbscan/hdbscan_distance_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
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

using daal::internal::CpuType;
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;
using daal::services::internal::TArray;
using daal::services::internal::TArrayScalable;

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           algorithms::internal::PairwiseDistanceType pairwiseDistance,
                                                                           double minkowskiDegree, int clusterSelection, bool allowSingleCluster,
                                                                           double clusterSelectionEpsilon, size_t maxClusterSize, double alpha,
                                                                           size_t leafSize)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();

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

    // Label output is stored as `int32_t` in the assignments NumericTable
    // (codebase-wide DAAL convention shared with kmeans / knn /
    // decision_forest / etc.). Refuse inputs where the label count could
    // exceed the destination-type bound. The label count is bounded above by
    // the number of surviving clusters, itself bounded by `nRows / mcs`.
    // Guard against `INT32_MAX` (not `INT_MAX`) because the storage type is
    // fixed-width `int32_t` regardless of the data model -- on a hypothetical
    // ILP64 platform `INT_MAX` would be 2^63 - 1 and would let overflowing
    // inputs through.
    if (nRows / minClusterSize > static_cast<size_t>(INT32_MAX))
    {
        return services::Status(services::ErrorIncorrectSizeOfInputNumericTable);
    }

    const size_t edgeCount = nRows - 1;

    ReadRows<algorithmFPType, cpu> dataBlock(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataBlock);
    const algorithmFPType * data = dataBlock.get();

    // =========================================================================
    // Step 1: Compute pairwise distance matrix
    // =========================================================================

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nRows);
    TArrayScalable<algorithmFPType, cpu> distMatrixVec(nRows * nRows);
    algorithmFPType * distMatrix = distMatrixVec.get();
    DAAL_CHECK_MALLOC(distMatrix);

    using algorithms::internal::PairwiseDistanceType;
    using algorithms::internal::EuclideanDistances;
    using algorithms::internal::CosineDistances;

    // For Euclidean and Cosine, reuse the shared GEMM-based primitives in
    // service_kernel_math.h. They are the same primitives knn uses, with a
    // blocked row-norm computation, A * A^T via xxgemm, and a blockwise
    // finalize() (SIMD max(0, .) clamp + batched vSqrt on 512-entry blocks)
    // that turns squared L2 into real L2 in place. For other metrics,
    // fillFullDistMatrix routes the row-pair loop through the corresponding
    // *Dist functor in hdbscan_distance_utils.h, so all three legacy inline
    // blocks (manhattan/chebyshev/minkowski) collapse to one call site driven
    // by the metric tag.
    if (pairwiseDistance == PairwiseDistanceType::euclidean)
    {
        // We need real (not squared) L2 here. Distances flow into three
        // downstream steps as actual distance values, not squared:
        //   - Core distances (Step 2) are selected from this matrix directly and
        //     become MST edge weights via
        //     MRD(a,b) = max(core(a), core(b), dist(a,b) / alpha).
        //   - `alpha` divides the pairwise dist term inside MRD; that scaling
        //     is only meaningful on real distances.
        //   - `clusterSelectionEpsilon` in sortMstAndExtractClusters is an
        //     actual distance threshold applied to MST edge weights, so an
        //     all-squared matrix would silently make the user's epsilon
        //     behave as if they had passed epsilon^2.
        //
        // Path: computeFull(squared=true) fills the matrix with squared L2
        // via A*A^T + row-norm expansion, then finalize() sweeps in 512-entry
        // blocks applying max(0, .) per entry (FP round-off on the diagonal /
        // near-duplicate rows can push squared values slightly below 0) and
        // a batched MathInst::vSqrt on the block in place. The
        // `squared=false` fast path is not used: its inline vSqrt has no
        // clamp and would yield NaN on those slightly-negative entries. The
        // diagonal is force-zeroed afterwards for defensive safety.
        EuclideanDistances<algorithmFPType, cpu> dist(*ntData, *ntData, /*squared=*/true);
        DAAL_CHECK_STATUS_VAR(dist.init());
        DAAL_CHECK_STATUS_VAR(dist.computeFull(distMatrix));
        DAAL_CHECK_STATUS_VAR(dist.finalize(nRows * nRows, distMatrix));
        for (size_t i = 0; i < nRows; i++) distMatrix[i * nRows + i] = algorithmFPType(0);
    }
    else if (pairwiseDistance == PairwiseDistanceType::cosine)
    {
        CosineDistances<algorithmFPType, cpu> dist(*ntData, *ntData);
        DAAL_CHECK_STATUS_VAR(dist.init());
        DAAL_CHECK_STATUS_VAR(dist.computeFull(distMatrix));
        // Zero-norm rows make 1 - dot/(aa*bb) divide by zero; FP round-off can
        // also push the diagonal away from a clean zero. Defensive cleanup.
        for (size_t i = 0; i < nRows; i++) distMatrix[i * nRows + i] = algorithmFPType(0);
    }
    else if (pairwiseDistance == PairwiseDistanceType::manhattan)
    {
        ManhattanDist<algorithmFPType> mh;
        fillFullDistMatrix<algorithmFPType, cpu>(data, nRows, nCols, mh, distMatrix);
    }
    else if (pairwiseDistance == PairwiseDistanceType::chebyshev)
    {
        ChebyshevDist<algorithmFPType> ch;
        fillFullDistMatrix<algorithmFPType, cpu>(data, nRows, nCols, ch, distMatrix);
    }
    else if (pairwiseDistance == PairwiseDistanceType::minkowski)
    {
        MinkowskiDist<algorithmFPType> mk(minkowskiDegree);
        fillFullDistMatrix<algorithmFPType, cpu>(data, nRows, nCols, mk, distMatrix);
    }
    else
    {
        // Unknown metric tag: fail loudly rather than silently routing to
        // one of the branches above.
        return services::Status(services::ErrorMethodNotSupported);
    }

    // =========================================================================
    // Step 2: Compute core distances (k-th nearest neighbor distance per point)
    //
    // Note: alpha scaling is applied later, only to dist(a,b) inside MRD
    // (Step 3). Per the canonical HDBSCAN definition, core distances must be
    // derived from the unscaled pairwise distance matrix.
    // =========================================================================

    TArray<algorithmFPType, cpu> coreDistsVec(nRows);
    algorithmFPType * coreDistances = coreDistsVec.get();
    DAAL_CHECK_MALLOC(coreDistances);

    // Core distance is the distance to the `minSamples`-th nearest neighbor in
    // the canonical HDBSCAN definition (Campello 2013), counting the point
    // itself as neighbor #1. The dense path sorts the full pairwise row, where
    // self is at index 0, so the answer sits at index `minSamples - 1`. Heap-
    // based paths (kd/ball-tree, GPU) use a different convention because their
    // top-k structure includes self as one of the k entries; do not align this
    // index with theirs.
    const size_t target = (minSamples > 0) ? minSamples - 1 : 0;
    const size_t t      = (target >= nRows) ? nRows - 1 : target;

    {
        // The core distance is the `t + 1`-th smallest entry of row `i`, so a
        // bounded max-heap of exactly that capacity is enough: `kthSmallestBounded`
        // streams the row once and keeps only `t + 1` floats of scratch per
        // thread. The earlier formulation copied the whole `nRows`-long row into
        // a per-thread `TlsMem(nRows)` slot and ran `std::nth_element` over it --
        // `2 * nRows` of extra traffic per point (`nRows^2` overall, 160 GB of
        // read + write at 100k float64) and `nThreads * nRows` of scratch, for
        // the same value. Selection is by value, so the result is unchanged.
        const size_t heapSize = t + 1;
        daal::TlsMem<algorithmFPType, cpu> tlsHeap(heapSize);
        SafeStatus safeStat;

        daal::threader_for(nRows, nRows, [&](size_t i) {
            algorithmFPType * heapBuf = tlsHeap.local();
            DAAL_CHECK_MALLOC_THR(heapBuf);

            coreDistances[i] = kthSmallestBounded<algorithmFPType, cpu>(distMatrix + i * nRows, nRows, heapSize, heapBuf);
        });

        DAAL_CHECK_SAFE_STATUS();
    }

    // =========================================================================
    // Step 3: Build MST using Boruvka's algorithm with MRD
    //
    // Prim's algorithm was tried here and reverted. On a complete graph Prim's
    // does strictly less total work -- `nRows - 1` iterations each streaming one
    // row, so one pass over the matrix against Boruvka's ~log2(N) -- but the two
    // differ in parallel shape, and on this path the shape dominates. Boruvka's
    // per-round query is a single `threader_for(nRows)` over independent rows:
    // N^2 of work behind one barrier, spread over every thread. Prim's needs N
    // sequential barriers with only N candidate slots to divide between them, so
    // past a few dozen blocks the task dispatch costs more than the scan and the
    // machine sits idle. Measured on a 224-thread host over the brute-force
    // benchmark set (15k-30k rows, 3-3072 features), block-partitioned Prim's
    // came out 20-25% slower than Boruvka on every case.
    //
    // What Boruvka is left paying is one full pass over the matrix per round.
    // The nearest-neighbour candidate cache built below removes most of those
    // passes without changing a single label; see the note on it.
    // =========================================================================

    TArray<DAAL_INT, cpu> mstFromVec(edgeCount);
    TArray<DAAL_INT, cpu> mstToVec(edgeCount);
    TArray<algorithmFPType, cpu> mstWeightsVec(edgeCount);
    DAAL_INT * mstFrom           = mstFromVec.get();
    DAAL_INT * mstTo             = mstToVec.get();
    algorithmFPType * mstWeights = mstWeightsVec.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    {
        TArray<DAAL_INT, cpu> ufParentVec(nRows);
        TArray<DAAL_INT, cpu> ufRankVec(nRows);
        TArray<DAAL_INT, cpu> componentOfVec(nRows);
        DAAL_INT * ufParent    = ufParentVec.get();
        DAAL_INT * ufRank      = ufRankVec.get();
        DAAL_INT * componentOf = componentOfVec.get();
        DAAL_CHECK_MALLOC(ufParent);
        DAAL_CHECK_MALLOC(ufRank);
        DAAL_CHECK_MALLOC(componentOf);

        TArray<algorithmFPType, cpu> pointBestMrdVec(nRows);
        TArray<DAAL_INT, cpu> pointBestIdxVec(nRows);
        algorithmFPType * pointBestMrd = pointBestMrdVec.get();
        DAAL_INT * pointBestIdx        = pointBestIdxVec.get();
        DAAL_CHECK_MALLOC(pointBestMrd);
        DAAL_CHECK_MALLOC(pointBestIdx);

        TArray<algorithmFPType, cpu> compBestMrdVec(nRows);
        TArray<DAAL_INT, cpu> compBestFromVec(nRows);
        TArray<DAAL_INT, cpu> compBestToVec(nRows);
        algorithmFPType * compBestMrd = compBestMrdVec.get();
        DAAL_INT * compBestFrom       = compBestFromVec.get();
        DAAL_INT * compBestTo         = compBestToVec.get();
        DAAL_CHECK_MALLOC(compBestMrd);
        DAAL_CHECK_MALLOC(compBestFrom);
        DAAL_CHECK_MALLOC(compBestTo);

        for (size_t i = 0; i < nRows; i++)
        {
            ufParent[i]    = static_cast<DAAL_INT>(i);
            componentOf[i] = static_cast<DAAL_INT>(i);
        }
        services::internal::service_memset_seq<DAAL_INT, cpu>(ufRank, 0, nRows);

        UnionFind uf { ufParent, ufRank };

        size_t edgesAdded    = 0;
        size_t numComponents = nRows;

        // Robust single linkage: scale only the pairwise dist term inside MRD
        // (canonical HDBSCAN), not the full distance matrix or core distances.
        // MRD(a, b) = max(core(a), core(b), dist(a, b) / alpha)
        const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
        const algorithmFPType inf      = daal::services::internal::MaxVal<algorithmFPType>::get();

        // A per-point cache of the `k` smallest MRD neighbours was tried here to
        // answer phase 1 without touching the matrix. It is exactly
        // label-preserving -- MRD is fixed once core distances are known, and the
        // first cached entry outside the point's component is provably the
        // row-wide `(MRD, index)` argmin -- but it measured only 1.03x geomean on
        // the brute-force benchmark set: Boruvka converges in very few rounds, so
        // there are few matrix passes to save, and building the cache costs one of
        // them. Not worth nRows * k * 16 bytes and the extra hot-path branch.
        while (numComponents > 1)
        {
            // Phase 1: For each point, find nearest different-component neighbor under MRD.
            // Only phase 1 is method-specific; phases 2-4 route through the shared
            // helpers in hdbscan_boruvka_utils.h.
            daal::threader_for(nRows, nRows, [&](size_t i) {
                const DAAL_INT myComp          = componentOf[i];
                const algorithmFPType * mrdRow = distMatrix + i * nRows;
                const algorithmFPType coreI    = coreDistances[i];
                algorithmFPType bestMrd        = inf;
                DAAL_INT bestIdx               = -1;

                for (size_t j = 0; j < nRows; j++)
                {
                    if (componentOf[j] == myComp) continue;
                    algorithmFPType mrd = mrdRow[j] * invAlpha;
                    mrd                 = (coreI > mrd) ? coreI : mrd;
                    mrd                 = (coreDistances[j] > mrd) ? coreDistances[j] : mrd;
                    if (mrd < bestMrd)
                    {
                        bestMrd = mrd;
                        bestIdx = static_cast<DAAL_INT>(j);
                    }
                }
                pointBestMrd[i] = bestMrd;
                pointBestIdx[i] = bestIdx;
            });

            reduceComponentBestEdges<algorithmFPType>(nRows, componentOf, pointBestMrd, pointBestIdx, compBestMrd, compBestFrom, compBestTo);

            const size_t addedThisRound = mergeComponentsEmitEdges<algorithmFPType>(nRows, compBestMrd, compBestFrom, compBestTo, uf, mstFrom, mstTo,
                                                                                    mstWeights, edgesAdded, numComponents);

            if (addedThisRound == 0) break;

            refreshComponentIds<cpu>(nRows, uf, componentOf);
        }

        // The MRD graph over a dense distance matrix is complete, so a round can
        // only fail to find a candidate when the matrix holds NaNs -- every
        // comparison against them is false -- i.e. on non-finite input. Bail out
        // rather than handing `sortMstAndExtractClusters` a truncated MST whose
        // tail entries of `mstFrom` / `mstTo` / `mstWeights` were never written:
        // those uninitialized indices are read as tree nodes and produce garbage
        // labels or an out-of-bounds access.
        if (edgesAdded != edgeCount) return services::Status(services::ErrorIncorrectInputNumericTable);
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters
    // =========================================================================

    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                       clusterSelection, allowSingleCluster, clusterSelectionEpsilon, maxClusterSize);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
