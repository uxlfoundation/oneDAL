/* file: kmeans_lloyd_distr_step2_impl.i */
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

#include "src/algorithms/kmeans/kmeans_lloyd_impl.i"

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
#define __DAAL_FABS(a) (((a) > (algorithmFPType)0.0) ? (a) : (-(a)))

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep2Kernel<method, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    const size_t nClusters = par->nClusters;
    const size_t p         = r[1]->getNumberOfColumns();
    int result             = 0;

    WriteOnlyRows<int, cpu> mtClusterS0(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS0);
    /* TODO: That should be size_t or double */
    int * clusterS0 = mtClusterS0.get();
    WriteOnlyRows<algorithmFPType, cpu> mtClusterS1(*const_cast<NumericTable *>(r[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS1);
    algorithmFPType * clusterS1 = mtClusterS1.get();
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunc(*const_cast<NumericTable *>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunc);
    algorithmFPType * goalFunc = mtTargetFunc.get();

    // partialCandidatesDistances is a 2-column table: col 0 = distance,
    // col 1 = source cluster id on the emitting rank. See
    // kmeans_partialresult.h for the layout rationale.
    WriteOnlyRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable *>(r[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    algorithmFPType * cValuesTbl = mtCValues.get();
    WriteOnlyRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable *>(r[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);
    algorithmFPType * cCentroids = mtCCentroids.get();

    const size_t nBlocks = na / 5;

    service_memset_seq<int, cpu>(clusterS0, 0, nClusters);
    service_memset_seq<algorithmFPType, cpu>(clusterS1, 0.0, nClusters * p);

    goalFunc[0] = 0;

    // Both columns of every slot are marked empty with -1: a negative distance in
    // column 0 is what the merge below and finalizeCompute treat as "no candidate
    // in this slot".
    service_memset_seq<algorithmFPType, cpu>(cValuesTbl, -1.0, nClusters * 2);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(size_t));

    // Merge tracks value (distance), the merged candidate slot id
    // `block * nClusters + posInBlock`, and the source cluster id on that
    // emitting rank. cCentroids in the output table is populated at the end
    // of this method from the winning candidate rows.
    TArray<algorithmFPType, cpu> tmpValues(nClusters);
    TArray<size_t, cpu> tmpIndices(nClusters);
    TArray<int, cpu> tmpSources(nClusters);
    TArray<size_t, cpu> cIndices(nClusters);
    DAAL_CHECK_MALLOC(tmpValues.get() && tmpIndices.get() && tmpSources.get() && cIndices.get());

    // Running merge state lives in cValuesTbl: col 0 (distance) and col 1
    // (source cluster id). Convenience aliases for the two columns.
    auto cValueAt  = [&](size_t idx) -> algorithmFPType & { return cValuesTbl[idx * 2 + 0]; };
    auto cSourceAt = [&](size_t idx) -> algorithmFPType & { return cValuesTbl[idx * 2 + 1]; };

    for (size_t i = 0; i < nBlocks; i++)
    {
        ReadRows<int, cpu> mtInClusterS0(*const_cast<NumericTable *>(a[i * 5 + 0]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
        ReadRows<algorithmFPType, cpu> mtInClusterS1(*const_cast<NumericTable *>(a[i * 5 + 1]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);
        ReadRows<algorithmFPType, cpu> mtInTargetFunc(*const_cast<NumericTable *>(a[i * 5 + 2]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);
        ReadRows<algorithmFPType, cpu> mtInCValues(*const_cast<NumericTable *>(a[i * 5 + 3]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInCValues);

        const int * inClusterS0              = mtInClusterS0.get();
        const algorithmFPType * inClusterS1  = mtInClusterS1.get();
        const algorithmFPType * inTargetFunc = mtInTargetFunc.get();
        const algorithmFPType * inCValues    = mtInCValues.get();

        for (size_t j = 0; j < nClusters; j++)
        {
            clusterS0[j] += inClusterS0[j];
        }

        for (size_t j = 0; j < nClusters * p; j++)
        {
            clusterS1[j] += inClusterS1[j];
        }

        goalFunc[0] += inTargetFunc[0];

        size_t cPos = 0, clPos = 0, cNum = 0;
        while (cNum < nClusters)
        {
            const algorithmFPType curDist = cValueAt(cPos);
            const algorithmFPType inDist  = inCValues[clPos * 2 + 0];
            if (curDist < (algorithmFPType)0.0 && inDist < (algorithmFPType)0.0)
            {
                break;
            }
            if (curDist > inDist)
            {
                tmpValues[cNum]  = curDist;
                tmpIndices[cNum] = cIndices[cPos];
                tmpSources[cNum] = static_cast<int>(cSourceAt(cPos));
                cNum++;
                cPos++;
            }
            else
            {
                tmpValues[cNum]  = inDist;
                tmpIndices[cNum] = i * nClusters + clPos;
                tmpSources[cNum] = static_cast<int>(inCValues[clPos * 2 + 1]);
                cNum++;
                clPos++;
            }
        }
        // Snapshot the merged top-K back into the running cValuesTbl / cIndices.
        for (size_t k = 0; k < cNum; k++)
        {
            cValueAt(k)  = tmpValues[k];
            cSourceAt(k) = static_cast<algorithmFPType>(tmpSources[k]);
        }
        result |= daal::services::internal::daal_memcpy_s(cIndices.get(), cNum * sizeof(size_t), tmpIndices.get(), cNum * sizeof(size_t));
    }

    for (size_t i = 0; i < nClusters; i++)
    {
        if (cValueAt(i) < (algorithmFPType)0.0)
        {
            break;
        }
        size_t block      = cIndices[i] / nClusters;
        size_t posInBlock = cIndices[i] % nClusters;

        ReadRows<algorithmFPType, cpu> mtInCCentroids(*const_cast<NumericTable *>(a[block * 5 + 4]), posInBlock, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInCCentroids);
        const algorithmFPType * inCCentroids = mtInCCentroids.get();
        result |= daal::services::internal::daal_memcpy_s(&cCentroids[i * p], p * sizeof(algorithmFPType), inCCentroids, p * sizeof(algorithmFPType));
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep2Kernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                   const NumericTable * const * r, const Parameter * par)
{
    const size_t p         = a[1]->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;

    ReadRows<int, cpu> mtInClusterS0(*const_cast<NumericTable *>(a[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
    ReadRows<algorithmFPType, cpu> mtInClusterS1(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);

    // The aggregated clusterS0 / clusterS1 arriving from step2::compute() are
    // needed as mutable working buffers: when a candidate row is used to seed
    // an empty cluster we have to subtract its contribution from the source
    // cluster it was assigned to. The input tables stay read-only (they belong
    // to the caller's partial result and finalizeCompute must be repeatable),
    // so work on private copies. This mirrors the two-pass empty-cluster
    // replacement logic in kmeans_lloyd_batch_impl.i (see the "Two-pass merge"
    // comment there).
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(int));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, p);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters * p, sizeof(algorithmFPType));

    TArray<int, cpu> clusterS0Arr(nClusters);
    TArray<algorithmFPType, cpu> clusterS1Arr(nClusters * p);
    DAAL_CHECK_MALLOC(clusterS0Arr.get() && clusterS1Arr.get());
    int * clusterS0             = clusterS0Arr.get();
    algorithmFPType * clusterS1 = clusterS1Arr.get();
    result |= daal::services::internal::daal_memcpy_s(clusterS0, nClusters * sizeof(int), mtInClusterS0.get(), nClusters * sizeof(int));
    result |= daal::services::internal::daal_memcpy_s(clusterS1, nClusters * p * sizeof(algorithmFPType), mtInClusterS1.get(),
                                                      nClusters * p * sizeof(algorithmFPType));
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    ReadRows<algorithmFPType, cpu> mtInTargetFunc(*const_cast<NumericTable *>(a[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);
    const algorithmFPType * inTarget = mtInTargetFunc.get();

    ReadRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable *>(a[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    ReadRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable *>(a[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);

    const algorithmFPType * cValuesTbl = mtCValues.get(); // 2 columns: [dist, srcCluster]
    const algorithmFPType * cCentroids = mtCCentroids.get();

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunct(*const_cast<NumericTable *>(r[1]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunct);

    algorithmFPType * clusters  = mtClusters.get();
    algorithmFPType * outTarget = mtTargetFunct.get();

    *outTarget = *inTarget;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(bool));
    TArray<bool, cpu> clusterReplaced(nClusters);
    DAAL_CHECK_MALLOC(clusterReplaced.get());
    service_memset_seq<bool, cpu>(clusterReplaced.get(), false, nClusters);

    // Two-pass merge (mirror of kmeans_lloyd_batch_impl.i):
    //   Pass 1: for every empty cluster i, promote the next-farthest candidate
    //           row (a row of cCentroids) to its centroid. Candidates arrive
    //           ordered by decreasing distance, so each empty cluster is handed
    //           the globally farthest-from-its-centroid point still available --
    //           the same purely distance-based selection scikit-learn's
    //           `_relocate_empty_clusters` uses. The candidate was originally
    //           assigned to some source cluster srcCluster on its emitting rank;
    //           that cluster's clusterS0/clusterS1 already carries the row's
    //           contribution after step2::compute() aggregation, so undo it now
    //           (when the source can still spare the row) and pass 2 computes
    //           the source cluster's centroid without the stolen row.
    //   Pass 2: normalize the remaining non-empty, non-replaced clusters.
    //   Pass 3: fill the clusters that are still empty (pass 1 stopped before
    //           them, or pass 1 stole all of their rows) with a duplicate of the
    //           centroid of the cluster holding the most observations.
    //
    // Candidates occupy slots [0, cAvail) of cValuesTbl / cCentroids ordered by
    // decreasing distance; slots with a negative distance are empty. `cNext` is
    // the cursor into that list.
    size_t cAvail = 0;
    while (cAvail < nClusters && !(cValuesTbl[cAvail * 2 + 0] < (algorithmFPType)0.0))
    {
        cAvail++;
    }

    // Seeds cluster `i` from candidate slot `c`: promotes the candidate row to
    // the cluster's centroid and undoes the row's contribution to the cluster it
    // was assigned to, so pass 2 computes that cluster's centroid without it.
    auto seedFromCandidate = [&](size_t i, size_t c) -> void {
        outTarget[0] -= cValuesTbl[c * 2 + 0];

        const int srcCluster            = static_cast<int>(cValuesTbl[c * 2 + 1]);
        const algorithmFPType * candRow = &cCentroids[c * p];
        result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), candRow, p * sizeof(algorithmFPType));

        if (srcCluster >= 0 && static_cast<size_t>(srcCluster) < nClusters && clusterS0[srcCluster] > 0)
        {
            clusterS0[srcCluster]--;
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[srcCluster * p + j] -= candRow[j];
            }
        }

        clusterReplaced[i] = true;
    };

    size_t cNext = 0;

    // Snapshot the empty clusters before pass 1 starts moving rows: pass 1
    // decrements the counters of the clusters it steals from, and a cluster it
    // drains that way must not be re-seeded from yet another candidate.
    // scikit-learn snapshots it for the same reason (`empty_clusters =
    // np.where(weight_in_clusters == 0)`).
    TArray<size_t, cpu> emptyClusters(nClusters);
    DAAL_CHECK_MALLOC(emptyClusters.get());
    size_t nEmpty = 0;
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] == 0)
        {
            emptyClusters[nEmpty] = i;
            nEmpty++;
        }
    }

    // Pass 1: seed the empty clusters from the candidates, farthest first.
    for (size_t e = 0; e < nEmpty; e++)
    {
        // Stop relocating once the farthest remaining candidate already sits on
        // the centroid it is assigned to: moving it leaves its source cluster's
        // mean unchanged and only plants a second centroid on top of an existing
        // one, which lets the assignment step flip labels between two identical
        // centroids forever. Candidates are sorted by decreasing distance, so all
        // later ones are at distance zero too. Running out of candidates means
        // the data holds fewer distinct rows than clusters and is handled the
        // same way: pass 3 fills whatever is left.
        if (cNext == cAvail || !(cValuesTbl[cNext * 2 + 0] > (algorithmFPType)0.0))
        {
            break;
        }
        seedFromCandidate(emptyClusters[e], cNext);
        cNext++;
    }

    // Pass 2: normalize the non-empty, non-replaced clusters. `largestCluster` is
    // the index of the one holding the most observations after pass 1's thefts,
    // or nClusters while no cluster holds any.
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

        for (size_t j = 0; j < p; j++)
        {
            clusters[i * p + j] = clusterS1[i * p + j] * coeff;
        }
    }

    // Pass 3. A cluster reaching this point holds no points and was not seeded by
    // pass 1: it was either empty on entry with no candidate left to relocate, or
    // pass 1 stole all of its rows. Fill it with a duplicate of the centroid of
    // the cluster holding the most observations, as the batch kernel does. The
    // data then holds fewer distinct rows than there are clusters, which is not an
    // error condition: scikit-learn reports it as a ConvergenceWarning ("Number of
    // distinct clusters found smaller than n_clusters") and returns a degenerate
    // centroid rather than failing. daal kernels have no warning channel, so the
    // closest available behaviour is to return the degenerate result too.
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterReplaced[i] || clusterS0[i] > 0)
        {
            continue;
        }
        if (largestCluster < nClusters)
        {
            result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &clusters[largestCluster * p],
                                                              p * sizeof(algorithmFPType));
        }
        else
        {
            // Not a single cluster holds an observation, so there is no centroid
            // to duplicate. Mirror scikit-learn's `_average_centers`, which leaves
            // a zero-weight cluster at its accumulated sum instead of dividing by
            // zero. `clusters` is write-only here, so it has to be written.
            result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &clusterS1[i * p],
                                                              p * sizeof(algorithmFPType));
        }
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
