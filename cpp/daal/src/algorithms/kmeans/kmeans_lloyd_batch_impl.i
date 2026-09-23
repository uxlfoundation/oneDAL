/* file: kmeans_lloyd_batch_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//++
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_defines.h"

#include "src/algorithms/kmeans/kmeans_lloyd_impl.i"
#include "src/algorithms/kmeans/kmeans_lloyd_postprocessing.h"

#include "src/services/service_profiler.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
/// Indices of the clusters that hold no points. Snapshotted before pass 1 starts
/// moving rows: pass 1 drains the counters of the clusters it steals from, and a
/// cluster drained that way must not be seeded again. scikit-learn's
/// `_relocate_empty_clusters` snapshots for the same reason.
template <CpuType cpu>
static size_t collectEmptyClusters(const size_t nClusters, const int * const clusterS0, size_t * const emptyClusters)
{
    size_t nEmpty = 0;
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] == 0)
        {
            emptyClusters[nEmpty] = i;
            nEmpty++;
        }
    }
    return nEmpty;
}

/// Seeds empty clusters with candidate rows farthest from their assigned centroids, 
/// mimicking scikit-learn's `_relocate_empty_clusters` logic.
///
/// Iterates through candidates sorted by decreasing distance, stopping when candidates 
/// are exhausted or distance reaches zero (to prevent duplicate centroids). Relocated 
/// rows are subtracted from their source cluster's aggregates.
///
/// @param ntData             Input data table of size `n x p`
/// @param p                  Number of features in the input data table
/// @param nClusters          Number of clusters
/// @param emptyClusters      Array of size `nClusters`; emptyClusters[i] stores the index of the i-th empty cluster
/// @param nEmpty             Number of empty clusters
/// @param cValues            Sorted array of maximum distances from assigned centroids to candidate rows
/// @param cIndices           Array of original dataset row indices corresponding to `cValues` candidates
/// @param cNum               Total number of available candidate rows in `cValues` and `cIndices`
/// @param pointAssignments   Array tracking the current cluster assignment index for each data row
/// @param inClusters         Array of size `nClusters x p` containing the initial centroid values
/// @param clusterS0          Output/Input array tracking number of points assigned per cluster (S0 statistic)
/// @param clusterS1          Output/Input array tracking the coordinate sum of points per cluster (S1 statistic)
/// @param clusters           Output array of size `nClusters x p` holding updated centroid coordinates
/// @param clusterReplaced    Output boolean array tracking which empty clusters have been successfully seeded
/// @param l2Norm             Output accumulator for the cumulative L2 norm of the relocated points
/// @param goalFuncCorrection Accumulates distances of relocated rows (measured against 
///                           initial centroids) for the caller to adjust the objective function.
template <typename algorithmFPType, CpuType cpu>
static Status relocateEmptyClusters(NumericTable * const ntData, const size_t p, const size_t nClusters, const size_t * const emptyClusters,
                                    const size_t nEmpty, const algorithmFPType * const cValues, const size_t * const cIndices, const size_t cNum,
                                    const int * const pointAssignments, const algorithmFPType * const inClusters, int * const clusterS0,
                                    algorithmFPType * const clusterS1, algorithmFPType * const clusters, bool * const clusterReplaced,
                                    algorithmFPType & l2Norm, algorithmFPType & goalFuncCorrection)
{
    size_t cPos = 0;
    for (size_t e = 0; e < nEmpty; e++)
    {
        if (cPos == cNum || !(cValues[cPos] > (algorithmFPType)0.0))
        {
            break;
        }

        const size_t i               = emptyClusters[e];
        const size_t candidateRowIdx = cIndices[cPos];
        goalFuncCorrection += cValues[cPos];
        cPos++;

        ReadRows<algorithmFPType, cpu> mtRow(ntData, candidateRowIdx, 1);
        DAAL_CHECK_BLOCK_STATUS(mtRow);
        const algorithmFPType * const row = mtRow.get();

        // The `> 0` test only keeps the counter from going negative: candidate
        // rows are distinct and each contributes exactly 1 to its source cluster.
        const int srcCluster = pointAssignments[candidateRowIdx];
        DAAL_ASSERT(srcCluster >= 0 && (size_t)srcCluster < nClusters);
        if (clusterS0[srcCluster] > 0)
        {
            clusterS0[srcCluster]--;
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[srcCluster * p + j] -= row[j];
            }
        }

        // Reduced into a per-cluster local so the inner loop can carry a
        // `reduction` clause; the partial is added to `l2Norm` once.
        algorithmFPType clusterL2Norm = (algorithmFPType)0.0;
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : clusterL2Norm))
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < p; j++)
        {
            const algorithmFPType dist = inClusters[i * p + j] - row[j];
            clusterL2Norm += dist * dist;
        }
        l2Norm += clusterL2Norm;

        DAAL_CHECK(!daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType)),
                   services::ErrorMemoryCopyFailedInternal);
        clusterReplaced[i] = true;
    }
    return Status();
}

/// Pass 2: recomputes the centroid of every cluster pass 1 did not seed, from the
/// theft-adjusted aggregates. Returns the cluster holding the most points, the
/// one pass 3 duplicates, or `nClusters` if no cluster holds any.
///
/// @param nClusters       Number of clusters
/// @param p               Number of features
/// @param clusterS0       Input array of size `nClusters` tracking the point count per cluster (S0 statistic)
/// @param clusterS1       Input array of size `nClusters x p` tracking the coordinate sum of points per cluster (S1 statistic)
/// @param clusterReplaced Input boolean array of size `nClusters` indicating which clusters were already seeded in Pass 1
/// @param inClusters      Input array of size `nClusters x p` containing the initial centroid coordinates
/// @param clusters        Output array of size `nClusters x p` holding the newly recomputed centroid coordinates
/// @param l2Norm          Output accumulator for the cumulative L2 norm distance between initial and updated centroids
/// @return                Index of the cluster holding the most points, or `nClusters` if all clusters are empty

template <typename algorithmFPType, CpuType cpu>
static size_t updateCentroidsFromAggregates(const size_t nClusters, const size_t p, const int * const clusterS0,
                                            const algorithmFPType * const clusterS1, const bool * const clusterReplaced,
                                            const algorithmFPType * const inClusters, algorithmFPType * const clusters, algorithmFPType & l2Norm)
{
    size_t largestCluster = nClusters;
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterReplaced[i] || clusterS0[i] <= 0)
        {
            continue;
        }
        if (largestCluster == nClusters || clusterS0[i] > clusterS0[largestCluster])
        {
            largestCluster = i;
        }

        const algorithmFPType coeff = (algorithmFPType)1.0 / clusterS0[i];

        algorithmFPType clusterL2Norm = (algorithmFPType)0.0;
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : clusterL2Norm))
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < p; j++)
        {
            const algorithmFPType newCluster = clusterS1[i * p + j] * coeff;
            const algorithmFPType dist       = inClusters[i * p + j] - newCluster;
            clusterL2Norm += dist * dist;
            clusters[i * p + j] = newCluster;
        }
        l2Norm += clusterL2Norm;
    }
    return largestCluster;
}

/// Fills remaining empty or fully-drained clusters with a duplicate of the 
/// largest cluster's centroid to maintain compatibility with scikit-learn.
///
/// If no clusters contain data points, the previous (initial) centroids are 
/// preserved to prevent uninitialized outputs.
///
/// @param nClusters       Number of clusters
/// @param p               Number of features in the cluster coordinate space
/// @param largestCluster  Index of the cluster holding the most points, used as the source for duplication
/// @param clusterS0       Input array of size `nClusters` tracking the point count per cluster (S0 statistic)
/// @param clusterReplaced Input boolean array of size `nClusters` indicating which clusters were already seeded in Pass 1
/// @param inClusters      Input array of size `nClusters x p` containing the initial centroid coordinates
/// @param clusters        Output array of size `nClusters x p` holding the updated centroid coordinates
/// @param l2Norm          Output accumulator for the cumulative L2 norm distance between initial and updated centroids
/// @return                Status of the execution

template <typename algorithmFPType, CpuType cpu>
static Status fillDrainedClusters(const size_t nClusters, const size_t p, const size_t largestCluster, const int * const clusterS0,
                                  const bool * const clusterReplaced, const algorithmFPType * const inClusters, algorithmFPType * const clusters,
                                  algorithmFPType & l2Norm)
{
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterReplaced[i] || clusterS0[i] > 0)
        {
            continue;
        }
        if (largestCluster < nClusters)
        {
            algorithmFPType clusterL2Norm = (algorithmFPType)0.0;
            PRAGMA_OMP_SIMD_ARGS(reduction(+ : clusterL2Norm))
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < p; j++)
            {
                const algorithmFPType newCluster = clusters[largestCluster * p + j];
                const algorithmFPType dist       = inClusters[i * p + j] - newCluster;
                clusterL2Norm += dist * dist;
                clusters[i * p + j] = newCluster;
            }
            l2Norm += clusterL2Norm;
        }
        else if (clusters != inClusters)
        {
            // Not a single cluster holds an observation, so there is no centroid
            // to duplicate. Keep the previous one, which also keeps `clusters[i]`
            // from being left uninitialized on iteration 0.
            DAAL_CHECK(!daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &inClusters[i * p],
                                                                p * sizeof(algorithmFPType)),
                       services::ErrorMemoryCopyFailedInternal);
        }
    }
    return Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansBatchKernel<method, algorithmFPType, cpu>::compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par)
{
    Status s;
    NumericTable * ntData  = const_cast<NumericTable *>(a[0]);
    const size_t nIter     = par->maxIterations;
    const size_t n         = ntData->getNumberOfRows();
    const size_t p         = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    // Cluster indices are narrowed to `int` when they are written into the
    // assignment table (`WriteOnlyRows<int, cpu>`) and into the internal
    // `pointAssignments` buffer, so they have to fit into `int`.
    DAAL_CHECK(nClusters <= static_cast<size_t>(services::internal::MaxVal<int>::get()), services::ErrorKMeansNumberOfClustersIsTooLarge);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(int));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(int));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, p);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters * p, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(algorithmFPType));

    TArray<int, cpu> clusterS0(nClusters);
    TArray<algorithmFPType, cpu> clusterS1(nClusters * p);
    TArray<bool, cpu> clusterReplaced(nClusters);
    TArray<size_t, cpu> emptyClusters(nClusters);
    // Per-point cluster assignment, needed only when a cluster ends up empty: the
    // row that seeds it is taken away from the cluster it was assigned to.
    TArray<int, cpu> pointAssignmentsHolder(n);
    DAAL_CHECK(clusterS0.get() && clusterS1.get() && clusterReplaced.get() && emptyClusters.get() && pointAssignmentsHolder.get(),
               services::ErrorMemoryAllocationFailed);
    int * pointAssignments = pointAssignmentsHolder.get();

    ReadRows<algorithmFPType, cpu> mtInClusters(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusters);
    algorithmFPType * inClusters = const_cast<algorithmFPType *>(mtInClusters.get());

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    algorithmFPType * clusters = mtClusters.get();

    TArray<algorithmFPType, cpu> tClusters;
    if (clusters == nullptr && nIter != 0)
    {
        tClusters.reset(nClusters * p);
        clusters = tClusters.get();
    }

    NumericTable * assignmetsNT = nullptr;
    NumericTablePtr assignmentsPtr;
    if (r[1])
    {
        assignmetsNT = const_cast<NumericTable *>(r[1]);
    }
    else if (par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        assignmentsPtr = HomogenNumericTableCPU<int, cpu>::create(1, n, &s);
        DAAL_CHECK_MALLOC(s);
        assignmetsNT = assignmentsPtr.get();
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(double));

    TArray<double, cpu> dS1(method == defaultDense ? p : 0);
    if (method == defaultDense)
    {
        DAAL_CHECK(dS1.get(), services::ErrorMemoryAllocationFailed);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(size_t));

    TArray<algorithmFPType, cpu> cValues(nClusters);
    TArray<size_t, cpu> cIndices(nClusters);
    // Not read here (batch resolves the source cluster from `pointAssignments`);
    // the buffer exists so batch and distributed share the same candidate search.
    TArray<int, cpu> cSources(nClusters);

    algorithmFPType oldTargetFunc(0.0);

    size_t blockSize = 0;
    DAAL_SAFE_CPU_CALL((blockSize = BSHelper<method, algorithmFPType, cpu>::kmeansGetBlockSize(n, p, nClusters)), (blockSize = 512))

    size_t kIter;

    for (kIter = 0; kIter < nIter; kIter++)
    {
        auto task = TaskKMeansLloyd<algorithmFPType, cpu>::create(p, nClusters, inClusters, blockSize);
        DAAL_CHECK(task.get(), services::ErrorMemoryAllocationFailed);
        {
            DAAL_PROFILER_TASK(addNTToTaskThreaded);
            /* For the last iteration we do not need to recount of assignmets */
            s = task->template addNTToTaskThreaded<method>(ntData, nullptr, blockSize, assignmetsNT && (kIter == nIter - 1) ? assignmetsNT : nullptr,
                                                           pointAssignments);
        }

        if (!s)
        {
            task->kmeansClearClusters(&oldTargetFunc);
            break;
        }

        {
            DAAL_PROFILER_TASK(kmeansPartialReduceCentroids);
            task->template kmeansComputeCentroids<method>(clusterS0.get(), clusterS1.get(), dS1.get());
        }

        size_t cNum;
        DAAL_CHECK_STATUS(s, task->kmeansComputeCentroidsCandidates(cValues.get(), cIndices.get(), cSources.get(), cNum));

        // Distances of the rows relocated into empty clusters, subtracted from the
        // objective function this iteration reports.
        algorithmFPType newCentersGoalFunc = (algorithmFPType)0.0;
        // Squared L2 norm of the centroid shift, `sum_i ||inClusters[i] - clusters[i]||^2`,
        // checked against `par->accuracyThreshold` below. Every cluster contributes
        // exactly once, whichever pass writes it; a cluster left at its previous
        // centroid shifts by zero.
        algorithmFPType l2Norm = (algorithmFPType)0.0;
        {
            DAAL_PROFILER_TASK(kmeansMergeReduceCentroids);

            // Empty clusters are handled in the three passes defined above this
            // function. Previous centroids are read from `inClusters` rather than
            // from `clusters`: on iteration 0 `clusters` is the write-only result
            // table and its contents are undefined, and from iteration 1 onward the
            // two alias.
            service_memset_seq<bool, cpu>(clusterReplaced.get(), false, nClusters);
            const size_t nEmpty = collectEmptyClusters<cpu>(nClusters, clusterS0.get(), emptyClusters.get());

            DAAL_CHECK_STATUS(s, (relocateEmptyClusters<algorithmFPType, cpu>(
                                     ntData, p, nClusters, emptyClusters.get(), nEmpty, cValues.get(), cIndices.get(), cNum, pointAssignments,
                                     inClusters, clusterS0.get(), clusterS1.get(), clusters, clusterReplaced.get(), l2Norm, newCentersGoalFunc)));

            const size_t largestCluster = updateCentroidsFromAggregates<algorithmFPType, cpu>(nClusters, p, clusterS0.get(), clusterS1.get(),
                                                                                              clusterReplaced.get(), inClusters, clusters, l2Norm);

            DAAL_CHECK_STATUS(s, (fillDrainedClusters<algorithmFPType, cpu>(nClusters, p, largestCluster, clusterS0.get(), clusterReplaced.get(),
                                                                            inClusters, clusters, l2Norm)));
        }
        {
            DAAL_PROFILER_TASK(kmeansUpdateObjectiveFunction);
            if (par->accuracyThreshold >= (algorithmFPType)0.0)
            {
                algorithmFPType newTargetFunc = (algorithmFPType)0.0;

                task->kmeansClearClusters(&newTargetFunc);
                newTargetFunc -= newCentersGoalFunc;

                if (l2Norm <= par->accuracyThreshold)
                {
                    kIter++;
                    break;
                }

                oldTargetFunc = newTargetFunc;
            }
            else
            {
                task->kmeansClearClusters(&oldTargetFunc);
                oldTargetFunc -= newCentersGoalFunc;
            }
        }
        inClusters = clusters;
    }

    if (!nIter)
    {
        clusters = inClusters;
    }

    if (par->resultsToEvaluate & computeAssignments || par->assignFlag || par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        PostProcessing<method, algorithmFPType, cpu>::computeAssignments(p, nClusters, clusters, ntData, nullptr, assignmetsNT, blockSize);
    }

    if (par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        WriteOnlyRows<algorithmFPType, cpu> mtTarget(*const_cast<NumericTable *>(r[2]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtTarget);
        algorithmFPType exactTargetFunc = algorithmFPType(0);
        PostProcessing<method, algorithmFPType, cpu>::computeExactObjectiveFunction(p, nClusters, clusters, ntData, nullptr, assignmetsNT,
                                                                                    exactTargetFunc, blockSize);

        *mtTarget.get() = exactTargetFunc;
    }
    if (r[3])
    {
        WriteOnlyRows<int, cpu> mtIterations(*const_cast<NumericTable *>(r[3]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtIterations);
        *mtIterations.get() = kIter;
    }
    return s;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
