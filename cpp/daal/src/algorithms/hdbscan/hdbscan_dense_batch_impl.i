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
 * HDBSCAN brute-force implementation:
 *   1. Full pairwise distance matrix
 *   2. Core distances: the minSamples-th smallest entry of each row
 *   3. Boruvka MST under MRD = max(coreI, coreJ, dist) / alpha, as in scikit-learn's brute path
 *   4. Sort the MST and extract clusters (shared with the tree methods)
 *
 * O(N^2) time and memory for the distance matrix.
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
services::Status HDBSCANBatchKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable * ntData, NumericTable * ntAssignments, NumericTable * ntNClusters, NumericTable * ntProbabilities,
    NumericTable * ntSingleLinkageTree, size_t minClusterSize, size_t minSamples, algorithms::internal::PairwiseDistanceType pairwiseDistance,
    double minkowskiDegree, int clusterSelection, bool allowSingleCluster, double clusterSelectionEpsilon, size_t maxClusterSize, double alpha,
    size_t leafSize)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();

    if (nRows < 2 || minClusterSize < 2)
    {
        WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(assignBlock);
        int * assignments = assignBlock.get();
        services::internal::service_memset<int, cpu>(assignments, -1, nRows);

        WriteOnlyRows<algorithmFPType, cpu> probBlock;
        algorithmFPType * probabilities = probBlock.set(ntProbabilities, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(probBlock);
        if (probabilities)
        {
            services::internal::service_memset<algorithmFPType, cpu>(probabilities, algorithmFPType(0), nRows);
        }

        WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(ncBlock);
        ncBlock.get()[0] = 0;
        return services::Status();
    }

    // Labels are stored as int32, and there are at most nRows / minClusterSize clusters.
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

    // Euclidean and cosine use the GEMM-based PairwiseDistances; the other metrics use the
    // functors in hdbscan_distance_utils.h.
    if (pairwiseDistance == PairwiseDistanceType::euclidean)
    {
        // MRD, alpha and clusterSelectionEpsilon all need real L2. finalize() clamps the squared
        // values at 0 before the root, which squared=false does not, so round-off cannot give NaN.
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
        // A zero-norm row makes 1 - dot/(aa*bb) divide by zero. scikit-learn
        // normalizes it to the zero vector instead, which puts it at distance 1
        // from every other row; round-off can also move the diagonal off zero.
        for (size_t i = 0; i < nRows; i++)
        {
            const algorithmFPType * row = data + i * nCols;
            bool isZero                 = true;
            for (size_t d = 0; d < nCols && isZero; d++) isZero = (row[d] == algorithmFPType(0));
            if (isZero)
            {
                for (size_t j = 0; j < nRows; j++)
                {
                    distMatrix[i * nRows + j] = algorithmFPType(1);
                    distMatrix[j * nRows + i] = algorithmFPType(1);
                }
            }
        }
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
        return services::Status(services::ErrorMethodNotSupported);
    }

    // =========================================================================
    // Step 2: Core distances, from the unscaled distances (alpha is applied in step 3)
    // =========================================================================

    TArray<algorithmFPType, cpu> coreDistsVec(nRows);
    algorithmFPType * coreDistances = coreDistsVec.get();
    DAAL_CHECK_MALLOC(coreDistances);

    // The core distance is the distance to the minSamples-th nearest neighbor, counting the point
    // itself, i.e. the minSamples-th smallest entry of its row.
    const size_t target = (minSamples > 0) ? minSamples - 1 : 0;
    const size_t t      = (target >= nRows) ? nRows - 1 : target;

    {
        const size_t heapSize = t + 1;
        daal::TlsMem<algorithmFPType, cpu> tlsHeap(heapSize);
        SafeStatus safeStat;

        daal::threader_for(nRows, 1, [&](size_t i) {
            algorithmFPType * heapBuf = tlsHeap.local();
            DAAL_CHECK_MALLOC_THR(heapBuf);

            coreDistances[i] = kthSmallestBounded<algorithmFPType, cpu>(distMatrix + i * nRows, nRows, heapSize, heapBuf);
        });

        DAAL_CHECK_SAFE_STATUS();
    }

    // =========================================================================
    // Step 3: Boruvka MST under mutual reachability distance
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

        // As in scikit-learn's brute path, MRD(a, b) = max(core(a), core(b), dist(a, b)) / alpha;
        // the tree methods scale only the dist term.
        const algorithmFPType invAlpha = static_cast<algorithmFPType>(1.0 / alpha);
        const algorithmFPType inf      = daal::services::internal::MaxVal<algorithmFPType>::get();
        if (alpha != 1.0)
        {
            for (size_t i = 0; i < nRows; i++) coreDistances[i] *= invAlpha;
        }

        while (numComponents > 1)
        {
            // Nearest neighbor of each point outside its component, under MRD.
            daal::threader_for(nRows, 1, [&](size_t i) {
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

        services::Status mstStatus = checkMst(edgesAdded, edgeCount, mstWeights);
        DAAL_CHECK_STATUS_VAR(mstStatus);
    }

    // =========================================================================
    // Steps 4-5: Sort MST + Extract clusters
    // =========================================================================

    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);
    int * assignments = assignBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> probBlock;
    algorithmFPType * probabilities = probBlock.set(ntProbabilities, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(probBlock);

    WriteOnlyRows<algorithmFPType, cpu> sltBlock;
    algorithmFPType * singleLinkageTree = sltBlock.set(ntSingleLinkageTree, 0, edgeCount);
    DAAL_CHECK_BLOCK_STATUS(sltBlock);

    int labelCounter = sortMstAndExtractClusters<algorithmFPType, cpu>(mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignments,
                                                                       clusterSelection, allowSingleCluster, clusterSelectionEpsilon, maxClusterSize,
                                                                       probabilities, singleLinkageTree);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
