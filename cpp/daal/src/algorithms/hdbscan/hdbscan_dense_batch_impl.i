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
 *   2. Compute core distances (k-th nearest neighbor distance per point)
 *   3. Build MST under Mutual Reachability Distance using Boruvka's algorithm,
 *      applying 1/alpha only to the dist term inside MRD = max(coreI, coreJ, dist/alpha)
 *   4. Sort MST + extract clusters via condensed tree + EOM/leaf (shared code)
 *
 * Complexity: O(N^2) for distance matrix, O(N^2 * log N) worst case for Boruvka's MST.
 * Memory:     O(N^2) for the distance matrix.
 */

#include <algorithm>
#include <cmath>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/hdbscan/hdbscan_distance_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
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
    // blocked row-norm computation, A·Aᵀ via xxgemm, and vectorized finalize.
    // For other metrics, fillFullDistMatrix routes the row-pair loop through
    // the corresponding *Dist functor in hdbscan_distance_utils.h, so all three
    // legacy inline blocks (manhattan/chebyshev/minkowski) collapse to one
    // call site driven by the metric tag.
    if (pairwiseDistance == PairwiseDistanceType::euclidean)
    {
        // Compute squared L2 first, then clamp(0, .) + vSqrt ourselves. The
        // shared `squared=false` path inside computeBatch does the vSqrt without
        // the max(0, .) clamp that finalize() applies, so FP round-off on the
        // diagonal (and on near-duplicate rows) produces NaN. Routing through
        // squared=true + manual finalize keeps every entry well-defined and
        // forces the diagonal to a clean zero.
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
    else // minkowski
    {
        MinkowskiDist<algorithmFPType> mk(minkowskiDegree);
        fillFullDistMatrix<algorithmFPType, cpu>(data, nRows, nCols, mk, distMatrix);
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
        daal::TlsMem<algorithmFPType, cpu> tlsBuf(nRows);
        SafeStatus safeStat;

        daal::threader_for(nRows, nRows, [&](size_t i) {
            algorithmFPType * dists = tlsBuf.local();
            DAAL_CHECK_MALLOC_THR(dists);

            const algorithmFPType * row = distMatrix + i * nRows;
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nRows; j++)
            {
                dists[j] = row[j];
            }

            std::nth_element(dists, dists + t, dists + nRows);
            coreDistances[i] = dists[t];
        });

        DAAL_CHECK_SAFE_STATUS();
    }

    // =========================================================================
    // Step 3: Build MST using Boruvka's algorithm with MRD
    // =========================================================================

    TArray<int, cpu> mstFromVec(edgeCount);
    TArray<int, cpu> mstToVec(edgeCount);
    TArray<algorithmFPType, cpu> mstWeightsVec(edgeCount);
    int * mstFrom                = mstFromVec.get();
    int * mstTo                  = mstToVec.get();
    algorithmFPType * mstWeights = mstWeightsVec.get();
    DAAL_CHECK_MALLOC(mstFrom);
    DAAL_CHECK_MALLOC(mstTo);
    DAAL_CHECK_MALLOC(mstWeights);

    {
        TArray<int, cpu> ufParentVec(nRows);
        TArray<int, cpu> ufRankVec(nRows);
        TArray<int, cpu> componentOfVec(nRows);
        int * ufParent    = ufParentVec.get();
        int * ufRank      = ufRankVec.get();
        int * componentOf = componentOfVec.get();
        DAAL_CHECK_MALLOC(ufParent);
        DAAL_CHECK_MALLOC(ufRank);
        DAAL_CHECK_MALLOC(componentOf);

        TArray<algorithmFPType, cpu> pointBestMrdVec(nRows);
        TArray<int, cpu> pointBestIdxVec(nRows);
        algorithmFPType * pointBestMrd = pointBestMrdVec.get();
        int * pointBestIdx             = pointBestIdxVec.get();
        DAAL_CHECK_MALLOC(pointBestMrd);
        DAAL_CHECK_MALLOC(pointBestIdx);

        TArray<algorithmFPType, cpu> compBestMrdVec(nRows);
        TArray<int, cpu> compBestFromVec(nRows);
        TArray<int, cpu> compBestToVec(nRows);
        algorithmFPType * compBestMrd = compBestMrdVec.get();
        int * compBestFrom            = compBestFromVec.get();
        int * compBestTo              = compBestToVec.get();
        DAAL_CHECK_MALLOC(compBestMrd);
        DAAL_CHECK_MALLOC(compBestFrom);
        DAAL_CHECK_MALLOC(compBestTo);

        for (size_t i = 0; i < nRows; i++)
        {
            ufParent[i]    = static_cast<int>(i);
            componentOf[i] = static_cast<int>(i);
        }
        services::internal::service_memset_seq<int, cpu>(ufRank, 0, nRows);

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

        size_t edgesAdded    = 0;
        size_t numComponents = nRows;

        // Robust single linkage: scale only the pairwise dist term inside MRD
        // (canonical HDBSCAN), not the full distance matrix or core distances.
        // MRD(a, b) = max(core(a), core(b), dist(a, b) / alpha)
        const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);

        while (numComponents > 1)
        {
            // Phase 1: For each point, find nearest different-component neighbor under MRD
            daal::threader_for(nRows, nRows, [&](size_t i) {
                const int myComp               = componentOf[i];
                const algorithmFPType * mrdRow = distMatrix + i * nRows;
                const algorithmFPType coreI    = coreDistances[i];
                algorithmFPType bestMrd        = daal::services::internal::MaxVal<algorithmFPType>::get();
                int bestIdx                    = -1;

                for (size_t j = 0; j < nRows; j++)
                {
                    if (componentOf[j] == myComp) continue;
                    algorithmFPType mrd = mrdRow[j] * invAlpha;
                    mrd                 = (coreI > mrd) ? coreI : mrd;
                    mrd                 = (coreDistances[j] > mrd) ? coreDistances[j] : mrd;
                    if (mrd < bestMrd)
                    {
                        bestMrd = mrd;
                        bestIdx = static_cast<int>(j);
                    }
                }
                pointBestMrd[i] = bestMrd;
                pointBestIdx[i] = bestIdx;
            });

            // Phase 2: Reduce to per-component best edge
            for (size_t i = 0; i < nRows; i++)
            {
                compBestMrd[i]  = daal::services::internal::MaxVal<algorithmFPType>::get();
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

            // Phase 3: Merge components via best edges
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

            // Phase 4: Update component IDs (parallelized)
            daal::threader_for(nRows, nRows, [&](size_t i) { componentOf[i] = ufFind(static_cast<int>(i)); });
        }
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters (shared with kd_tree)
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
