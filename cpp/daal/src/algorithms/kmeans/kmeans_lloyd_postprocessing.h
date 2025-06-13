/* file: kmeans_lloyd_postprocessing.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#ifndef _KMEANS_LLOYD_POSTPROCESSING_H__
#define _KMEANS_LLOYD_POSTPROCESSING_H__

#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_defines.h"
#include "src/algorithms/service_error_handling.h"

#include "src/threading/threading.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_spblas.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
struct PostProcessing
{
    static Status computeAssignments(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters, const NumericTable * ntData,
                                     algorithmFPType * catCoef, NumericTable * ntAssign, const size_t blockSizeDefault);

    static Status computeExactObjectiveFunction(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters,
                                                const NumericTable * const ntData, const algorithmFPType * const catCoef, NumericTable * ntAssign,
                                                algorithmFPType & objectiveFunction, const size_t blockSizeDefault);
};

template <typename algorithmFPType, CpuType cpu>
struct PostProcessing<lloydDense, algorithmFPType, cpu>
{
    static Status computeAssignments(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters, const NumericTable * ntData,
                                     algorithmFPType * catCoef, NumericTable * ntAssign, const size_t blockSizeDefault)
    {
        const size_t n       = ntData->getNumberOfRows();
        const size_t nBlocks = n / blockSizeDefault + !!(n % blockSizeDefault);

        /* Allocate memory for all arrays inside TLS */
        daal::tls<algorithmFPType *> tlsTask([=]() { return service_scalable_malloc<algorithmFPType, cpu>(blockSizeDefault * nClusters); });

        TArrayScalable<algorithmFPType, cpu> clSq(nClusters);
        DAAL_CHECK(clSq.get(), services::ErrorMemoryAllocationFailed);

        for (size_t k = 0; k < nClusters; k++)
        {
            algorithmFPType sum = algorithmFPType(0);
            PRAGMA_FORCE_SIMD
            PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
            for (size_t j = 0; j < p; j++)
            {
                sum += inClusters[k * p + j] * inClusters[k * p + j] * 0.5;
            }
            clSq[k] = sum;
        }

        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [&](int iBlock) {
            algorithmFPType * x_clusters = tlsTask.local();
            DAAL_CHECK_MALLOC_THR(x_clusters);
            const size_t blockSize = (iBlock == nBlocks - 1) ? n - iBlock * blockSizeDefault : blockSizeDefault;

            ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable *>(ntData), iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(mtData);
            const algorithmFPType * const data = mtData.get();

            WriteOnlyRows<int, cpu> assignBlock(ntAssign, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            int * assignments = assignBlock.get();

            const algorithmFPType * clustersSq = clSq.get();
            char transa                        = 't';
            char transb                        = 'n';
            DAAL_INT _m                        = nClusters;
            DAAL_INT _n                        = blockSize;
            DAAL_INT _k                        = p;
            algorithmFPType alpha              = 1.0;
            DAAL_INT lda                       = p;
            DAAL_INT ldy                       = p;
            algorithmFPType beta               = 0.0;
            DAAL_INT ldaty                     = nClusters;

            BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, inClusters, &lda, data, &ldy, &beta, x_clusters, &ldaty);

            for (size_t i = 0; i < blockSize; i++)
            {
                algorithmFPType minGoalVal = clustersSq[0] - x_clusters[i * nClusters];
                size_t minIdx              = 0;

                for (size_t j = 1; j < nClusters; j++)
                {
                    if (minGoalVal > clustersSq[j] - x_clusters[i * nClusters + j])
                    {
                        minGoalVal = clustersSq[j] - x_clusters[i * nClusters + j];
                        minIdx     = j;
                    }
                }

                assignments[i] = minIdx;
            }
        });
        DAAL_CHECK_SAFE_STATUS();

        tlsTask.reduce([](algorithmFPType * array) { service_scalable_free<algorithmFPType, cpu>(array); });
        return Status();
    }

    static Status computeExactObjectiveFunction(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters,
                                                const NumericTable * const ntData, const algorithmFPType * const catCoef, NumericTable * ntAssign,
                                                algorithmFPType & objectiveFunction, const size_t blockSizeDefault)
    {
        const size_t n       = ntData->getNumberOfRows();
        const size_t nBlocks = n / blockSizeDefault + !!(n % blockSizeDefault);

        TArrayScalable<algorithmFPType, cpu> goalLocal(nBlocks);
        algorithmFPType * goalLocalData = goalLocal.get();
        DAAL_CHECK_MALLOC(goalLocalData);

        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [&](const int iBlock) {
            const size_t blockSize = (iBlock == nBlocks - 1) ? n - iBlock * blockSizeDefault : blockSizeDefault;

            ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable *>(ntData), iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(mtData);
            const algorithmFPType * const data = mtData.get();

            ReadRows<int, cpu> assignBlock(ntAssign, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            const int * assignments = assignBlock.get();

            algorithmFPType goal = algorithmFPType(0);
            for (size_t k = 0; k < blockSize; k++)
            {
                const size_t assk = assignments[k];
                PRAGMA_VECTOR_UNALIGNED
                for (size_t j = 0; j < p; j++)
                {
                    goal += (data[k * p + j] - inClusters[assk * p + j]) * (data[k * p + j] - inClusters[assk * p + j]);
                }
            } /* for (size_t k = 0; k < blockSize; k++) */
            goalLocalData[iBlock] = goal;
        }); /* daal::threader_for( nBlocks, nBlocks, [=](int k) */

        DAAL_CHECK_SAFE_STATUS();

        objectiveFunction = algorithmFPType(0);
        PRAGMA_VECTOR_UNALIGNED
        for (size_t j = 0; j < nBlocks; j++)
        {
            objectiveFunction += goalLocalData[j];
        }
        return Status();
    }
};

template <typename algorithmFPType, CpuType cpu>
struct PostProcessing<lloydCSR, algorithmFPType, cpu>
{
    static Status computeAssignments(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters, const NumericTable * ntData,
                                     algorithmFPType * catCoef, NumericTable * ntAssign, const size_t blockSizeDefault)
    {
        const size_t n       = ntData->getNumberOfRows();
        const size_t nBlocks = n / blockSizeDefault + !!(n % blockSizeDefault);

        /* Allocate memory for all arrays inside TLS */
        daal::tls<algorithmFPType *> tlsTask([=]() { return service_scalable_malloc<algorithmFPType, cpu>(blockSizeDefault * nClusters); });

        TArrayScalable<algorithmFPType, cpu> clSq(nClusters);
        DAAL_CHECK(clSq.get(), services::ErrorMemoryAllocationFailed);

        for (size_t k = 0; k < nClusters; k++)
        {
            clSq[k]             = 0;
            algorithmFPType sum = algorithmFPType(0);
            PRAGMA_FORCE_SIMD
            PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
            for (size_t j = 0; j < p; j++)
            {
                sum += inClusters[k * p + j] * inClusters[k * p + j] * 0.5;
            }
            clSq[k] = sum;
        }

        SafeStatus safeStat;
        CSRNumericTableIface * ntDataCsr = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntData));
        DAAL_CHECK(ntDataCsr, services::ErrorEmptyCSRNumericTable);

        daal::threader_for(nBlocks, nBlocks, [&](int iBlock) {
            algorithmFPType * x_clusters = tlsTask.local();
            DAAL_CHECK_MALLOC_THR(x_clusters);
            const size_t blockSize = (iBlock == nBlocks - 1) ? n - iBlock * blockSizeDefault : blockSizeDefault;

            ReadRowsCSR<algorithmFPType, cpu> dataBlock(ntDataCsr, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(dataBlock);

            const algorithmFPType * const data = dataBlock.values();
            const size_t * const colIdx        = dataBlock.cols();
            const size_t * const rowIdx        = dataBlock.rows();

            const algorithmFPType * clustersSq = clSq.get();

            WriteOnlyRows<int, cpu> assignBlock(ntAssign, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            int * assignments = assignBlock.get();

            const char transa           = 'n';
            const DAAL_INT _n           = blockSize;
            const DAAL_INT _p           = p;
            const DAAL_INT _c           = nClusters;
            const algorithmFPType alpha = 1.0;
            const algorithmFPType beta  = 0.0;
            const char matdescra[6]     = { 'G', 0, 0, 'F', 0, 0 };

            SpBlasInst<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra, data, (DAAL_INT *)colIdx, (DAAL_INT *)rowIdx,
                                                      inClusters, &_p, &beta, x_clusters, &_n);

            for (size_t i = 0; i < blockSize; i++)
            {
                algorithmFPType minGoalVal = clustersSq[0] - x_clusters[i];
                size_t minIdx              = 0;

                for (size_t j = 0; j < nClusters; j++)
                {
                    if (minGoalVal > clustersSq[j] - x_clusters[i + j * blockSize])
                    {
                        minGoalVal = clustersSq[j] - x_clusters[i + j * blockSize];
                        minIdx     = j;
                    }
                }
                assignments[i] = minIdx;
            }
        });
        DAAL_CHECK_SAFE_STATUS();
        tlsTask.reduce([](algorithmFPType * array) { service_scalable_free<algorithmFPType, cpu>(array); });
        return Status();
    }

    static Status computeExactObjectiveFunction(const size_t p, const size_t nClusters, const algorithmFPType * const inClusters,
                                                const NumericTable * const ntData, const algorithmFPType * const catCoef, NumericTable * ntAssign,
                                                algorithmFPType & objectiveFunction, const size_t blockSizeDefault)
    {
        const size_t n       = ntData->getNumberOfRows();
        const size_t nBlocks = n / blockSizeDefault + !!(n % blockSizeDefault);

        TArrayScalable<algorithmFPType, cpu> goalLocal(nBlocks);
        algorithmFPType * goalLocalData = goalLocal.get();
        DAAL_CHECK_MALLOC(goalLocalData);

        CSRNumericTableIface * ntDataCsr = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntData));
        DAAL_CHECK(ntDataCsr, services::ErrorEmptyCSRNumericTable);

        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [&](const int iBlock) {
            const size_t blockSize = (iBlock == nBlocks - 1) ? n - iBlock * blockSizeDefault : blockSizeDefault;

            ReadRowsCSR<algorithmFPType, cpu> dataBlock(ntDataCsr, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(dataBlock);

            ReadRows<int, cpu> assignBlock(ntAssign, iBlock * blockSizeDefault, blockSize);
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            const int * assignments = assignBlock.get();

            const algorithmFPType * const data = dataBlock.values();
            const size_t * const colIdx        = dataBlock.cols();
            const size_t * const rowIdx        = dataBlock.rows();

            algorithmFPType goal = algorithmFPType(0);
            for (size_t k = 0; k < blockSize; k++)
            {
                const size_t jStart  = rowIdx[k] - 1;
                const size_t jFinish = rowIdx[k + 1] - 1;

                const size_t assk = assignments[k];
                PRAGMA_VECTOR_UNALIGNED
                for (size_t j = jStart; j < jFinish; j++)
                {
                    const size_t m = colIdx[j] - 1;
                    goal += (data[j] - inClusters[assk * p + m]) * (data[j] - inClusters[assk * p + m]);
                }
            } /* for (size_t k = 0; k < blockSize; k++) */
            goalLocalData[iBlock] = goal;
        }); /* daal::threader_for( nBlocks, nBlocks, [=](int k) */
        DAAL_CHECK_SAFE_STATUS();

        objectiveFunction = algorithmFPType(0);
        PRAGMA_VECTOR_UNALIGNED
        for (size_t j = 0; j < nBlocks; j++)
        {
            objectiveFunction += goalLocalData[j];
        }

        return Status();
    }
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
