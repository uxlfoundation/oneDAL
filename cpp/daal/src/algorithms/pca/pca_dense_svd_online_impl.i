/* file: pca_dense_svd_online_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_SVD_ONLINE_IMPL_I__
#define __PCA_DENSE_SVD_ONLINE_IMPL_I__

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::internal;
using namespace daal::data_management;

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDOnlineKernel<algorithmFPType, cpu>::compute(InputDataType type, const NumericTablePtr & data, NumericTable & nObservations,
                                                                   NumericTable & auxiliaryTable, NumericTable & sumSVD, NumericTable & sumSquaresSVD)
{
    if (type == correlation) return services::Status(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);

    const size_t nVectors = data->getNumberOfRows();

    WriteRows<int, cpu> oldObservationsBlock(nObservations, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(oldObservationsBlock);
    int * oldObservations         = oldObservationsBlock.get();
    const size_t nOldObservations = *oldObservations;

    NumericTablePtr normalizedData;
    if (type == normalizedDataset)
    {
        normalizedData = data;
    }
    else
    {
        const size_t totalObservations = nOldObservations + nVectors;
        services::Status s             = normalizeDataset(data, totalObservations, nObservations, sumSVD, sumSquaresSVD, normalizedData);
        if (!s) return s;
    }
    services::Status s = decompose(normalizedData.get(), auxiliaryTable);
    *oldObservations += nVectors;
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDOnlineKernel<algorithmFPType, cpu>::finalizeMerge(InputDataType type, const NumericTablePtr & nObservationsTable,
                                                                         NumericTable & eigenvalues, NumericTable & eigenvectors,
                                                                         DataCollectionPtr & rTables)
{
    if (type == correlation) return services::Status(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);

    const int nVectors = nObservationsTable->getValue<int>(0, 0);
    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    const size_t np      = rTables->size();
    const size_t nInputs = np * 2;
    TArray<NumericTable *, cpu> svdInputs(nInputs);
    DAAL_CHECK_MALLOC(svdInputs.get());

    for (size_t i = 0; i < np; i++)
    {
        svdInputs[i]      = static_cast<NumericTable *>(rTables->get(i).get());
        svdInputs[i + np] = 0;
    }

    const size_t nResults               = 3;
    NumericTable * svdResults[nResults] = { &eigenvalues, nullptr, &eigenvectors };
    daal::algorithms::svd::internal::SVDOnlineKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    services::Status s = svdKernel.finalizeCompute(nInputs, svdInputs.get(), nResults, svdResults, &kmPar);
    if (s) s = this->scaleSingularValues(eigenvalues, nVectors);
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDOnlineKernel<algorithmFPType, cpu>::decompose(const NumericTable * normalizedDataTable, NumericTable & auxiliaryTable)
{
    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    const size_t na                        = 1;
    const NumericTable * const * svdInputs = &normalizedDataTable;

    const size_t nr               = 2;
    NumericTable * svdResults[nr] = { 0, &auxiliaryTable };

    daal::algorithms::svd::internal::SVDOnlineKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    return svdKernel.compute(na, svdInputs, nr, svdResults, &kmPar);
}

namespace
{
template <typename algorithmFPType, CpuType cpu>
inline void computeSumsAndSsq(const size_t nObservations, const size_t nFeatures, const algorithmFPType * data, algorithmFPType * sums,
                              algorithmFPType * ssq)
{
    for (size_t i = 0; i < nObservations; i++)
    {
        for (size_t j = 0; j < nFeatures; j++)
        {
            sums[j] += data[i * nFeatures + j];
            ssq[j] += data[i * nFeatures + j] * data[i * nFeatures + j];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
inline void computeMean(const size_t nObservations, const size_t nFeatures, const algorithmFPType * sums, algorithmFPType * mean)
{
    for (size_t j = 0; j < nFeatures; j++)
    {
        mean[j] = sums[j] / nObservations;
    }
}

template <typename algorithmFPType, CpuType cpu>
inline void computeVariance(const size_t nObservations, const size_t nFeatures, const algorithmFPType * sums, const algorithmFPType * ssq,
                            const algorithmFPType * mean, algorithmFPType * variance)
{
    for (size_t j = 0; j < nFeatures; j++)
    {
        variance[j] = daal::internal::MathInst<algorithmFPType, cpu>::sSqrt((ssq[j] - 2 * mean[j] * sums[j] + nObservations * mean[j] * mean[j])
                                                                            / (nObservations - 1));
    }
}

template <typename algorithmFPType, CpuType cpu>
inline void normalizeData(const size_t nObservations, const size_t nFeatures, const algorithmFPType * data, const algorithmFPType * mean,
                          const algorithmFPType * variance, algorithmFPType * normalizedData)
{
    for (size_t i = 0; i < nObservations; i++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            normalizedData[i * nFeatures + j] = (data[i * nFeatures + j] - mean[j]) / variance[j];
        }
    }
}
} // namespace

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDOnlineKernel<algorithmFPType, cpu>::normalizeDataset(const NumericTablePtr & data, size_t totalObservations,
                                                                            NumericTable & nObservations, NumericTable & sumSVD,
                                                                            NumericTable & sumSquaresSVD, NumericTablePtr & normalizedData)
{
    using daal::internal::HomogenNumericTableCPU;

    const size_t nVectors = data->getNumberOfRows();
    ReadRows<algorithmFPType, cpu> block(const_cast<NumericTable &>(*data.get()), 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(block);
    const algorithmFPType * dataArray = block.get();

    const size_t nFeatures = data->getNumberOfColumns();
    services::Status s;
    services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > normalized =
        HomogenNumericTableCPU<algorithmFPType, cpu>::create(nFeatures, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    normalized->assign(0);

    normalizedData = normalized;

    WriteOnlyRows<algorithmFPType, cpu> normalizedBlock(*normalizedData, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(normalizedBlock);
    algorithmFPType * normalizedDataArray = normalizedBlock.get();

    WriteRows<algorithmFPType, cpu> blockSums(sumSVD, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(blockSums);
    algorithmFPType * sums = blockSums.get();

    WriteRows<algorithmFPType, cpu> blockSsq(sumSquaresSVD, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(blockSsq);
    algorithmFPType * ssq = blockSsq.get();

    TArray<algorithmFPType, cpu> mean(nFeatures);
    TArray<algorithmFPType, cpu> variance(nFeatures);
    DAAL_CHECK_MALLOC(mean.get() && variance.get());
    computeSumsAndSsq<algorithmFPType, cpu>(nVectors, nFeatures, dataArray, sums, ssq);
    computeMean<algorithmFPType, cpu>(totalObservations, nFeatures, sums, mean.get());
    computeVariance<algorithmFPType, cpu>(totalObservations, nFeatures, sums, ssq, mean.get(), variance.get());
    normalizeData<algorithmFPType, cpu>(nVectors, nFeatures, dataArray, mean.get(), variance.get(), normalizedDataArray);
    return Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
