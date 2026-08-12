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

    /* TODO: initialization  */
    for (size_t j = 0; j < nClusters; j++)
    {
        clusterS0[j] = 0;
    }

    for (size_t j = 0; j < nClusters * p; j++)
    {
        clusterS1[j] = 0;
    }

    goalFunc[0] = 0;

    for (size_t j = 0; j < nClusters; j++)
    {
        cValuesTbl[j * 2 + 0] = (algorithmFPType)-1.0;
        cValuesTbl[j * 2 + 1] = (algorithmFPType)-1.0;
    }

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
    TArray<int, cpu> cSources(nClusters);
    DAAL_CHECK_MALLOC(tmpValues.get() && tmpIndices.get() && tmpSources.get() && cIndices.get() && cSources.get());

    // Running merge state stored in cValuesTbl col 0 (distance).
    // Convenience alias for the two columns.
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
                tmpSources[cNum] = cSources[cPos];
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
        // Snapshot the merged top-K back into the running cValuesTbl / cIndices / cSources.
        for (size_t k = 0; k < cNum; k++)
        {
            cValueAt(k)  = tmpValues[k];
            cSourceAt(k) = static_cast<algorithmFPType>(tmpSources[k]);
        }
        result |= daal::services::internal::daal_memcpy_s(cIndices.get(), cNum * sizeof(size_t), tmpIndices.get(), cNum * sizeof(size_t));
        result |= daal::services::internal::daal_memcpy_s(cSources.get(), cNum * sizeof(int), tmpSources.get(), cNum * sizeof(int));
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

    // The aggregated clusterS0 / clusterS1 arriving from step2::compute() are
    // treated here as mutable working buffers -- we may need to subtract off
    // the contribution of a candidate row that gets used to seed an empty
    // cluster. This mirrors the two-pass empty-cluster replacement logic in
    // kmeans_lloyd_batch_impl.i (see the "Two-pass merge" comment there). We
    // cannot use ReadRows for these because their rows must be mutable in
    // the theft-adjustment step.
    ReadWriteMode readWriteAggregates = readWrite;
    BlockDescriptor<int> blkClusterS0;
    BlockDescriptor<algorithmFPType> blkClusterS1;
    const_cast<NumericTable *>(a[0])->getBlockOfRows(0, nClusters, readWriteAggregates, blkClusterS0);
    const_cast<NumericTable *>(a[1])->getBlockOfRows(0, nClusters, readWriteAggregates, blkClusterS1);
    int * clusterS0             = blkClusterS0.getBlockPtr();
    algorithmFPType * clusterS1 = blkClusterS1.getBlockPtr();

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
    //   Pass 1: for every empty cluster i, promote the next unused candidate
    //           row (cCentroids[cPos]) to its centroid. The candidate was
    //           originally assigned to some source cluster srcCluster on its
    //           emitting rank; that cluster's clusterS0/clusterS1 already
    //           carries the row's contribution after step2::compute()
    //           aggregation. Undo it now so pass 2 computes the source
    //           cluster's centroid without the stolen row.
    //   Pass 2: normalize the remaining non-empty, non-replaced clusters.
    //           Corner case: a source cluster drained completely by pass 1
    //           has clusterS0[src]==0 in pass 2. In that case there is no
    //           previous-iteration centroid available to step2 (the master
    //           broadcasts it back only at the start of the next iteration),
    //           so we fall back to the drained cluster's original candidate
    //           row -- the same row we already promoted, which is guaranteed
    //           to have been in that cluster before the theft.
    size_t cPos = 0;

    // Pass 1: replace empties.
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] == 0)
        {
            DAAL_CHECK(!(cValuesTbl[cPos * 2 + 0] < (algorithmFPType)0.0), services::ErrorKMeansNumberOfClustersIsTooLarge);
            outTarget[0] -= cValuesTbl[cPos * 2 + 0];

            const int srcCluster            = static_cast<int>(cValuesTbl[cPos * 2 + 1]);
            const algorithmFPType * candRow = &cCentroids[cPos * p];
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
            cPos++;
        }
    }

    // Pass 2: normalize the non-empty, non-replaced clusters.
    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterReplaced[i])
        {
            continue;
        }
        if (clusterS0[i] > 0)
        {
            const algorithmFPType coeff = (algorithmFPType)1.0 / clusterS0[i];

            for (size_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = clusterS1[i * p + j] * coeff;
            }
        }
        else
        {
            // Source cluster fully drained by pass 1. Reuse the last-known
            // candidate row that was in this cluster before the theft; this
            // is the closest local approximation to the previous centroid
            // that finalizeCompute has visibility into.
            DAAL_CHECK(!(cValuesTbl[cPos * 2 + 0] < (algorithmFPType)0.0), services::ErrorKMeansNumberOfClustersIsTooLarge);
            result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &cCentroids[cPos * p],
                                                              p * sizeof(algorithmFPType));
            cPos++;
        }
    }

    const_cast<NumericTable *>(a[0])->releaseBlockOfRows(blkClusterS0);
    const_cast<NumericTable *>(a[1])->releaseBlockOfRows(blkClusterS1);

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
