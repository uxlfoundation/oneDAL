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
 *   1. Compute full pairwise distance matrix via GEMM (with alpha scaling)
 *   2. Compute core distances (k-th nearest neighbor distance per point)
 *   3. Build MST under Mutual Reachability Distance using Boruvka's algorithm
 *   4. Sort MST + extract clusters via condensed tree + EOM/leaf (shared code)
 *
 * Complexity: O(N^2) for distance matrix, O(N^2 * log N) worst case for Boruvka's MST.
 * Memory:     O(N^2) for the distance matrix.
 */

#include <algorithm>
#include <cmath>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"
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

using daal::internal::BlasInst;
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
    // Step 1: Compute pairwise distance matrix
    // =========================================================================

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nRows);
    TArrayScalable<algorithmFPType, cpu> distMatrixVec(nRows * nRows);
    algorithmFPType * distMatrix = distMatrixVec.get();
    DAAL_CHECK_MALLOC(distMatrix);

    using algorithms::internal::PairwiseDistanceType;

    if (pairwiseDistance == PairwiseDistanceType::euclidean || pairwiseDistance == PairwiseDistanceType::cosine)
    {
        // GEMM-accelerated path for Euclidean and Cosine
        TArray<algorithmFPType, cpu> normsVec(nRows);
        algorithmFPType * norms = normsVec.get();
        DAAL_CHECK_MALLOC(norms);

        const size_t normBlockSize = 512;
        const size_t nNormBlocks   = (nRows + normBlockSize - 1) / normBlockSize;

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

        // Compute A * A^T via GEMM
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

        const size_t distBlockSize = 256;
        const size_t nDistBlocks   = (nRows + distBlockSize - 1) / distBlockSize;

        if (pairwiseDistance == PairwiseDistanceType::euclidean)
        {
            // Convert dot products to Euclidean distances
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
        }
        else // cosine
        {
            // Convert dot products to cosine distances: 1 - dot / (||a|| * ||b||)
            daal::threader_for(nDistBlocks, nDistBlocks, [&](size_t iBlock) {
                const size_t i_begin = iBlock * distBlockSize;
                const size_t i_end   = (i_begin + distBlockSize > nRows) ? nRows : i_begin + distBlockSize;

                for (size_t i = i_begin; i < i_end; i++)
                {
                    algorithmFPType * row    = distMatrix + i * nRows;
                    const algorithmFPType ni = static_cast<algorithmFPType>(sqrt(static_cast<double>(norms[i])));
                    for (size_t j = 0; j < nRows; j++)
                    {
                        const algorithmFPType nj    = static_cast<algorithmFPType>(sqrt(static_cast<double>(norms[j])));
                        const algorithmFPType denom = ni * nj;
                        algorithmFPType d;
                        if (denom > algorithmFPType(0))
                            d = algorithmFPType(1) - row[j] / denom;
                        else
                            d = algorithmFPType(0);
                        if (d < algorithmFPType(0)) d = algorithmFPType(0);
                        row[j] = d;
                    }
                    row[i] = algorithmFPType(0);
                }
            });
        }
    }
    else if (pairwiseDistance == PairwiseDistanceType::manhattan)
    {
        const size_t rowBlockSize = 64;
        const size_t nRowBlocks   = (nRows + rowBlockSize - 1) / rowBlockSize;

        daal::threader_for(nRowBlocks, nRowBlocks, [&](size_t iBlock) {
            const size_t i_begin = iBlock * rowBlockSize;
            const size_t i_end   = (i_begin + rowBlockSize > nRows) ? nRows : i_begin + rowBlockSize;

            for (size_t i = i_begin; i < i_end; i++)
            {
                const algorithmFPType * row_i = data + i * nCols;
                algorithmFPType * dist_row    = distMatrix + i * nRows;

                for (size_t j = i; j < nRows; j++)
                {
                    const algorithmFPType * row_j = data + j * nCols;
                    algorithmFPType d             = algorithmFPType(0);
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t f = 0; f < nCols; f++)
                    {
                        algorithmFPType diff = row_i[f] - row_j[f];
                        d += (diff >= algorithmFPType(0)) ? diff : -diff;
                    }
                    dist_row[j]               = d;
                    distMatrix[j * nRows + i] = d;
                }
                dist_row[i] = algorithmFPType(0);
            }
        });
    }
    else if (pairwiseDistance == PairwiseDistanceType::chebyshev)
    {
        const size_t rowBlockSize = 64;
        const size_t nRowBlocks   = (nRows + rowBlockSize - 1) / rowBlockSize;

        daal::threader_for(nRowBlocks, nRowBlocks, [&](size_t iBlock) {
            const size_t i_begin = iBlock * rowBlockSize;
            const size_t i_end   = (i_begin + rowBlockSize > nRows) ? nRows : i_begin + rowBlockSize;

            for (size_t i = i_begin; i < i_end; i++)
            {
                const algorithmFPType * row_i = data + i * nCols;
                algorithmFPType * dist_row    = distMatrix + i * nRows;

                for (size_t j = i; j < nRows; j++)
                {
                    const algorithmFPType * row_j = data + j * nCols;
                    algorithmFPType d             = algorithmFPType(0);
                    for (size_t f = 0; f < nCols; f++)
                    {
                        algorithmFPType diff = row_i[f] - row_j[f];
                        if (diff < algorithmFPType(0)) diff = -diff;
                        if (diff > d) d = diff;
                    }
                    dist_row[j]               = d;
                    distMatrix[j * nRows + i] = d;
                }
                dist_row[i] = algorithmFPType(0);
            }
        });
    }
    else // minkowski
    {
        const double p            = minkowskiDegree;
        const double invp         = 1.0 / p;
        const size_t rowBlockSize = 64;
        const size_t nRowBlocks   = (nRows + rowBlockSize - 1) / rowBlockSize;

        daal::threader_for(nRowBlocks, nRowBlocks, [&](size_t iBlock) {
            const size_t i_begin = iBlock * rowBlockSize;
            const size_t i_end   = (i_begin + rowBlockSize > nRows) ? nRows : i_begin + rowBlockSize;

            for (size_t i = i_begin; i < i_end; i++)
            {
                const algorithmFPType * row_i = data + i * nCols;
                algorithmFPType * dist_row    = distMatrix + i * nRows;

                for (size_t j = i; j < nRows; j++)
                {
                    const algorithmFPType * row_j = data + j * nCols;
                    double dsum                   = 0.0;
                    for (size_t f = 0; f < nCols; f++)
                    {
                        double diff = static_cast<double>(row_i[f] - row_j[f]);
                        if (diff < 0.0) diff = -diff;
                        dsum += pow(diff, p);
                    }
                    const algorithmFPType d   = static_cast<algorithmFPType>(pow(dsum, invp));
                    dist_row[j]               = d;
                    distMatrix[j * nRows + i] = d;
                }
                dist_row[i] = algorithmFPType(0);
            }
        });
    }

    // =========================================================================
    // Step 1b: Apply alpha scaling (robust single linkage)
    // =========================================================================
    if (alpha != 1.0)
    {
        const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
        const size_t totalDist         = nRows * nRows;
        const size_t alphaBlockSize    = 4096;
        const size_t nAlphaBlocks      = (totalDist + alphaBlockSize - 1) / alphaBlockSize;

        daal::threader_for(nAlphaBlocks, nAlphaBlocks, [&](size_t iBlock) {
            const size_t begin = iBlock * alphaBlockSize;
            const size_t end   = (begin + alphaBlockSize > totalDist) ? totalDist : begin + alphaBlockSize;
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t idx = begin; idx < end; idx++)
            {
                distMatrix[idx] *= invAlpha;
            }
        });
    }

    // =========================================================================
    // Step 2: Compute core distances (k-th nearest neighbor distance per point)
    // =========================================================================

    TArray<algorithmFPType, cpu> coreDistsVec(nRows);
    algorithmFPType * coreDistances = coreDistsVec.get();
    DAAL_CHECK_MALLOC(coreDistances);

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
                    algorithmFPType mrd = mrdRow[j];
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
