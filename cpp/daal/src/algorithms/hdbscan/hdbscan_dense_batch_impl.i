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
 *   1. Compute full pairwise Euclidean distance matrix via GEMM
 *   2. Compute core distances (k-th nearest neighbor distance per point)
 *   3. Build MST under Mutual Reachability Distance using Prim's algorithm
 *   4. Sort MST + extract clusters via condensed tree + EOM (shared code)
 *
 * Complexity: O(N^2) for distance matrix and Prim's MST.
 * Memory:     O(N^2) for the distance matrix.
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
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"
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

using daal::internal::BlasInst;
using daal::internal::CpuType;
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(const NumericTable * ntData, NumericTable * ntAssignments,
                                                                           NumericTable * ntNClusters, size_t minClusterSize, size_t minSamples,
                                                                           int metric, double degree, int clusterSelection,
                                                                           bool allowSingleCluster)
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

    daal::services::internal::TArray<algorithmFPType, cpu> distMatrixArr(nRows * nRows);
    algorithmFPType * distMatrix = distMatrixArr.get();
    DAAL_CHECK_MALLOC(distMatrix);

    if (metric == euclidean || metric == cosine)
    {
        // GEMM-accelerated path for Euclidean and Cosine
        daal::services::internal::TArray<algorithmFPType, cpu> normsArr(nRows);
        algorithmFPType * norms = normsArr.get();
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

        if (metric == euclidean)
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
    else if (metric == manhattan)
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
    else if (metric == chebyshev)
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
        const double p            = degree;
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
    // Step 2: Compute core distances (k-th nearest neighbor distance per point)
    // =========================================================================

    daal::services::internal::TArray<algorithmFPType, cpu> coreDists(nRows);
    algorithmFPType * coreDistances = coreDists.get();
    DAAL_CHECK_MALLOC(coreDistances);

    const size_t target = (minSamples > 0) ? minSamples - 1 : 0;
    const size_t t      = (target >= nRows) ? nRows - 1 : target;

    {
        daal::TlsMem<algorithmFPType, cpu> tlsBuf(nRows);

        daal::threader_for(nRows, nRows, [&](size_t i) {
            algorithmFPType * dists = tlsBuf.local();
            if (!dists) return;

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
    }

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

    // Build MST using Prim's algorithm with MRD weights.
    // O(N^2) time using dense distance matrix, O(N) extra space.
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

        for (size_t i = 0; i < nRows; i++)
        {
            minEdge[i] = std::numeric_limits<algorithmFPType>::max();
            minFrom[i] = 0;
            inMst[i]   = 0;
        }

        // Start from node 0
        inMst[0]                        = 1;
        const algorithmFPType core0     = coreDistances[0];
        const algorithmFPType * distRow = distMatrix;
        for (size_t j = 1; j < nRows; j++)
        {
            algorithmFPType d = distRow[j];
            d                 = (core0 > d) ? core0 : d;
            d                 = (coreDistances[j] > d) ? coreDistances[j] : d;
            minEdge[j]        = d;
        }

        for (size_t e = 0; e < edgeCount; e++)
        {
            // Find minimum edge to non-MST node
            int best                = -1;
            algorithmFPType bestW   = std::numeric_limits<algorithmFPType>::max();
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

            // Update min_edge for remaining nodes
            const algorithmFPType * bestRow  = distMatrix + static_cast<size_t>(best) * nRows;
            const algorithmFPType coreBest   = coreDistances[best];
            for (size_t j = 0; j < nRows; j++)
            {
                if (inMst[j]) continue;
                algorithmFPType d = bestRow[j];
                d                 = (coreBest > d) ? coreBest : d;
                d                 = (coreDistances[j] > d) ? coreDistances[j] : d;
                if (d < minEdge[j])
                {
                    minEdge[j] = d;
                    minFrom[j] = best;
                }
            }
        }
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters (shared with kd_tree)
    // =========================================================================

    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                        clusterSelection, allowSingleCluster);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
