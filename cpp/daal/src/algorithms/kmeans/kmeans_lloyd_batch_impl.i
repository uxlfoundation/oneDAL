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
template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansBatchKernel<method, algorithmFPType, cpu>::compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par)
{
    Status s;
    NumericTable * ntData  = const_cast<NumericTable *>(a[0]);
    const size_t nIter     = par->maxIterations;
    const size_t n         = ntData->getNumberOfRows();
    const size_t p         = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;

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
    DAAL_CHECK(clusterS0.get() && clusterS1.get() && clusterReplaced.get() && emptyClusters.get(), services::ErrorMemoryAllocationFailed);

    // Per-point cluster assignment tracked across the outer Lloyd loop.
    // Needed only when at least one cluster ends up empty: the replacement
    // candidate steals a point from its previously-assigned cluster, and we
    // must decrement that cluster's counters (clusterS0 / clusterS1) before
    // computing its new centroid.
    TArray<int, cpu> pointAssignmentsHolder(n);
    DAAL_CHECK(pointAssignmentsHolder.get(), services::ErrorMemoryAllocationFailed);
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
    // Per-candidate source cluster IDs. Batch already resolves srcCluster via
    // pointAssignments[candidateRowIdx], so this buffer's contents are not read
    // here -- the parameter exists so batch and distributed can share
    // kmeansComputeCentroidsCandidates.
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
        size_t cPos = 0;

        algorithmFPType newCentersGoalFunc = (algorithmFPType)0.0;
        algorithmFPType l2Norm             = (algorithmFPType)0.0;
        service_memset_seq<bool, cpu>(clusterReplaced.get(), false, nClusters);
        {
            DAAL_PROFILER_TASK(kmeansMergeReduceCentroids);

            // Two-pass merge:
            //   Pass 1: replace each empty cluster's centroid with a candidate
            //           row -- the point that is farthest from the centroid it
            //           was assigned to on this iteration. This matches
            //           scikit-learn's `_relocate_empty_clusters`: the empty
            //           clusters take the globally farthest-from-their-centroid
            //           points, selected purely by distance regardless of how
            //           many points their source cluster holds. Each such point
            //           is stolen from the cluster it was assigned to, so
            //           decrement that source cluster's counters (clusterS0,
            //           clusterS1) here; pass 2 then computes the source
            //           cluster's centroid without the stolen row.
            //   Pass 2: compute centroids for the non-replaced clusters from the
            //           (now theft-adjusted) aggregates and accumulate the L2
            //           shift.
            //
            // Candidates are sorted by decreasing squared distance, so cPos
            // walks them farthest-first. Reading previous centroids from
            // `inClusters`, not from the write-only `clusters` buffer: on the
            // very first iteration `clusters` is the user-supplied result table
            // whose contents are uninitialized prior to being written. From
            // iteration 1 onward `inClusters == clusters` (set at the end of the
            // loop), so the L2-norm of the centroid shift is unchanged.
            //
            // The set of empty clusters is snapshotted before pass 1 starts
            // moving rows: pass 1 decrements the counters of the clusters it
            // steals from, and a cluster it drains that way must not be re-seeded
            // from yet another candidate. scikit-learn snapshots it for the same
            // reason (`empty_clusters = np.where(weight_in_clusters == 0)`).
            size_t nEmpty = 0;
            for (size_t i = 0; i < nClusters; i++)
            {
                if (clusterS0[i] == 0)
                {
                    emptyClusters[nEmpty] = i;
                    nEmpty++;
                }
            }

            for (size_t e = 0; e < nEmpty; e++)
            {
                DAAL_CHECK(cPos < cNum, services::ErrorKMeansNumberOfClustersIsTooLarge);
                // Stop relocating once the farthest remaining candidate already
                // sits on the centroid it is assigned to. Such a row cannot
                // improve the objective function: moving it out of its cluster
                // leaves that cluster's mean unchanged ((mean * n - mean) /
                // (n - 1) == mean) and only plants a second centroid on top of
                // an existing one. The duplicate then makes the assignment step
                // tie-break between two identical centroids, which can flip
                // labels from iteration to iteration without ever reducing the
                // objective function. Leaving the cluster at its previous
                // centroid keeps the result well defined and lets the L2-norm
                // check below terminate. All later candidates are at distance
                // zero too (the list is sorted), so the whole pass is done.
                if (!(cValues[cPos] > (algorithmFPType)0.0))
                {
                    break;
                }

                const size_t i = emptyClusters[e];
                newCentersGoalFunc += cValues[cPos];
                const size_t candidateRowIdx = cIndices[cPos];
                ReadRows<algorithmFPType, cpu> mtRow(ntData, candidateRowIdx, 1);
                const algorithmFPType * row = mtRow.get();

                // Take the candidate away from its currently-assigned cluster so
                // pass 2 computes that cluster's centroid without it. Several
                // candidates can come from the same source cluster (or a
                // candidate can be the only point of its cluster), which drains
                // the source to zero; the `> 0` guard keeps the counters
                // non-negative and pass 2 leaves a drained cluster at its
                // previous centroid.
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

                // Accumulated into a per-cluster partial first: adding the terms
                // straight into `l2Norm`, which already carries the shift of every
                // cluster processed so far, loses the low-order bits of each term.
                algorithmFPType clusterL2Norm = (algorithmFPType)0.0;
                PRAGMA_OMP_SIMD_ARGS(reduction(+ : clusterL2Norm))
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < p; j++)
                {
                    const algorithmFPType dist = inClusters[i * p + j] - row[j];
                    clusterL2Norm += dist * dist;
                }
                l2Norm += clusterL2Norm;

                result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
                clusterReplaced[i] = true;
                cPos++;
            }
            for (size_t i = 0; i < nClusters; i++)
            {
                if (clusterReplaced[i])
                {
                    continue;
                }
                if (clusterS0[i] > 0)
                {
                    const algorithmFPType coeff = 1.0 / clusterS0[i];

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
                else if (clusters != inClusters)
                {
                    // Cluster holds no points and pass 1 did not seed it. Either
                    // it was empty on entry and pass 1 stopped before reaching it
                    // (no candidate left that is not already on its own
                    // centroid), or it was non-empty on entry and pass 1 stole
                    // all of its rows. Both cases keep the previous centroid, so
                    // `clusters[i]` is never left uninitialized on iter 0 and the
                    // cluster contributes nothing to the L2 shift. Skipped when
                    // `clusters` and `inClusters` alias (iter >= 1 with tClusters
                    // unused), in which case the value is already there.
                    result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &inClusters[i * p],
                                                                      p * sizeof(algorithmFPType));
                }
            }
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
    return (!result) ? s : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
