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
    DAAL_CHECK(clusterS0.get() && clusterS1.get() && clusterReplaced.get(), services::ErrorMemoryAllocationFailed);

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
            //           row (the farthest-from-nearest-centroid point that
            //           kmeansComputeCentroidsCandidates produced). The
            //           candidate is stolen from the cluster it was assigned
            //           to on this iteration, so decrement that source
            //           cluster's counters (clusterS0, clusterS1) here so the
            //           subsequent centroid computation in pass 2 reflects
            //           the theft. Candidates whose source cluster holds a
            //           single point are passed over -- see the skip loop
            //           below.
            //   Pass 2: compute centroids for the originally-non-empty
            //           clusters from the (now theft-adjusted) aggregates and
            //           accumulate the L2 shift. A cluster that pass 1 left
            //           empty (no usable candidate) falls back to its previous
            //           centroid (inClusters) so `clusters[i * p + j]` is
            //           always defined on iter 0.
            //
            // Reading previous centroids from `inClusters`, not from the
            // write-only `clusters` buffer: on the very first iteration
            // `clusters` is the user-supplied result table whose contents
            // are uninitialized prior to being written. From iteration 1
            // onward `inClusters == clusters` (set at the end of the loop),
            // so the L2-norm of the centroid shift is unchanged.
            for (size_t i = 0; i < nClusters; i++)
            {
                if (clusterS0[i] == 0)
                {
                    DAAL_CHECK(cPos < cNum, services::ErrorKMeansNumberOfClustersIsTooLarge);

                    // Pass over candidates whose source cluster holds a single
                    // point: taking it would empty the source cluster, so the
                    // number of empty clusters would not go down, and the two
                    // centroids would end up on the same point. A skipped
                    // candidate is gone for good -- cluster sizes only shrink
                    // within this pass, so it cannot become usable later.
                    while (cPos < cNum && clusterS0[pointAssignments[cIndices[cPos]]] <= 1)
                    {
                        cPos++;
                    }
                    if (cPos == cNum)
                    {
                        // Every remaining candidate is the only point of its
                        // cluster. Leave cluster i empty for this iteration;
                        // pass 2 keeps its previous centroid.
                        continue;
                    }

                    newCentersGoalFunc += cValues[cPos];
                    const size_t candidateRowIdx = cIndices[cPos];
                    ReadRows<algorithmFPType, cpu> mtRow(ntData, candidateRowIdx, 1);
                    const algorithmFPType * row = mtRow.get();

                    // Take the candidate away from its currently-assigned cluster
                    // so pass 2 computes that cluster's centroid without it.
                    const int srcCluster = pointAssignments[candidateRowIdx];
                    DAAL_ASSERT(srcCluster >= 0 && (size_t)srcCluster < nClusters);
                    DAAL_ASSERT(clusterS0[srcCluster] > 1);
                    clusterS0[srcCluster]--;
                    PRAGMA_OMP_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; j++)
                    {
                        clusterS1[srcCluster * p + j] -= row[j];
                    }

                    PRAGMA_OMP_SIMD_ARGS(reduction(+ : l2Norm))
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; j++)
                    {
                        const algorithmFPType dist = inClusters[i * p + j] - row[j];
                        l2Norm += dist * dist;
                    }
                    result |=
                        daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
                    clusterReplaced[i] = true;
                    cPos++;
                }
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

                    PRAGMA_OMP_SIMD_ARGS(reduction(+ : l2Norm))
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; j++)
                    {
                        const algorithmFPType newCluster = clusterS1[i * p + j] * coeff;
                        const algorithmFPType dist       = inClusters[i * p + j] - newCluster;
                        l2Norm += dist * dist;
                        clusters[i * p + j] = newCluster;
                    }
                }
                else if (clusters != inClusters)
                {
                    // Cluster is empty and pass 1 found no usable candidate for
                    // it (every remaining candidate was the only point of its
                    // own cluster). Fall back to the previous centroid so we
                    // never leave `clusters[i]` uninitialized on iter 0. Note
                    // that pass 1 can no longer drain an originally non-empty
                    // cluster: it never takes the last point of a cluster.
                    // Skipped when
                    // `clusters` and `inClusters` alias (iter >= 1 with
                    // tClusters unused), in which case the value is already
                    // there.
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
