/* file: kmeans_lloyd_distr_step1_impl.i */
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
Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    NumericTable * ntData        = const_cast<NumericTable *>(a[0]);
    NumericTable * ntAssignments = const_cast<NumericTable *>(r[5]);

    const size_t n         = ntData->getNumberOfRows();
    const size_t p         = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;
    size_t blockSize       = 0;
    DAAL_SAFE_CPU_CALL((blockSize = BSHelper<method, algorithmFPType, cpu>::kmeansGetBlockSize(n, p, nClusters)), (blockSize = 512))

    ReadRows<algorithmFPType, cpu> mtInitClusters(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInitClusters);
    algorithmFPType * initClusters = const_cast<algorithmFPType *>(mtInitClusters.get());
    WriteOnlyRows<int, cpu> mtClusterS0(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS0);
    WriteOnlyRows<int, cpu> mtClusterS2(*const_cast<NumericTable *>(r[0]), 0, n);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS2)
    /* TODO: That should be size_t or double */
    int * clusterS0 = mtClusterS0.get();
    int * clusterS2 = mtClusterS2.get();
    WriteOnlyRows<algorithmFPType, cpu> mtClusterS1(*const_cast<NumericTable *>(r[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS1);
    algorithmFPType * clusterS1 = mtClusterS1.get();
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunc(*const_cast<NumericTable *>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunc);
    algorithmFPType * goalFunc = mtTargetFunc.get();

    WriteOnlyRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable *>(r[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    algorithmFPType * cValues = mtCValues.get();
    WriteOnlyRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable *>(r[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);
    algorithmFPType * cCentroids = mtCCentroids.get();

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    for (size_t i = 0; i < p; i++)
    {
        if (ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
        {
            catFlag = 1;
            break;
        }
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> catCoef(catFlag ? p : 0);
    if (catFlag)
    {
        DAAL_CHECK(catCoef.get(), services::ErrorMemoryAllocationFailed);
        for (size_t i = 0; i < p; i++)
        {
            if (ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
            {
                catCoef[i] = par->gamma;
            }
            else
            {
                catCoef[i] = (algorithmFPType)1.0;
            }
        }
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(size_t));

    TArray<size_t, cpu> cIndices(nClusters);
    DAAL_CHECK_MALLOC(cIndices.get());

    Status s;
    algorithmFPType oldTargetFunc = (algorithmFPType)0.0;
    {
        auto task = TaskKMeansLloyd<algorithmFPType, cpu>::create(p, nClusters, n, initClusters, blockSize);
        DAAL_CHECK(task.get(), services::ErrorMemoryAllocationFailed);
        DAAL_ASSERT(task);

        if (par->resultsToEvaluate & computeAssignments || par->assignFlag)
        {
            s = task->template addNTToTaskThreaded<method>(ntData, catCoef.get(), blockSize, ntAssignments);
        }
        else
        {
            s = task->template addNTToTaskThreaded<method>(ntData, catCoef.get(), blockSize);
        }
        if (!s)
        {
            task->kmeansClearClusters(&oldTargetFunc);
            return s;
        }

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(double));

        TArray<double, cpu> dS1(method == defaultDense ? p : 0);
        if (method == defaultDense)
        {
            DAAL_CHECK(dS1.get(), services::ErrorMemoryAllocationFailed);
        }

        task->template kmeansComputeCentroids<method>(clusterS0, clusterS2, clusterS1, dS1.get());

        size_t cNum;
        DAAL_CHECK_STATUS(s, task->kmeansComputeCentroidsCandidates(cValues, cIndices.get(), cNum));
        for (size_t i = 0; i < cNum; i++)
        {
            ReadRows<algorithmFPType, cpu> mtRow(ntData, cIndices.get()[i], 1);
            const algorithmFPType * row = mtRow.get();
            result |= daal::services::internal::daal_memcpy_s(&cCentroids[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
        }
        for (size_t i = cNum; i < nClusters; i++)
        {
            cValues[i] = (algorithmFPType)-1.0;
        }

        task->kmeansClearClusters(goalFunc);
    }
    return (!result) ? s : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                   const NumericTable * const * r, const Parameter * par)
{
    if (!(par->resultsToEvaluate & computeAssignments || par->assignFlag)) return Status();

    NumericTable * ntPartialAssignments = const_cast<NumericTable *>(a[0]);
    NumericTable * ntAssignments        = const_cast<NumericTable *>(r[0]);
    const size_t n                      = ntPartialAssignments->getNumberOfRows();

    ReadRows<int, cpu> inBlock(*ntPartialAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(inBlock);
    const int * inAssignments = inBlock.get();

    WriteOnlyRows<int, cpu> outBlock(*ntAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(outBlock);
    int * outAssignments = outBlock.get();

    PRAGMA_IVDEP
    for (size_t i = 0; i < n; i++)
    {
        outAssignments[i] = inAssignments[i];
    }
    return Status();
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
