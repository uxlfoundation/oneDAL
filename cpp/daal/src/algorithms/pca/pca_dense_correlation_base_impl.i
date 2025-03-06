/* file: pca_dense_correlation_base_impl.i */
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

#include "src/algorithms/pca/pca_dense_correlation_base.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::signFlipEigenvectors(NumericTable & eigenvectors) const
{
    return PCADenseBase<algorithmFPType, cpu>::signFlipEigenvectors(eigenvectors);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::fillTable(NumericTable & table, algorithmFPType val) const
{
    return PCADenseBase<algorithmFPType, cpu>::fillTable(table, val);
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationBase<algorithmFPType, cpu>::copyArray(size_t size, const algorithmFPType * source, algorithmFPType * destination)
{
    if (source != destination)
    {
        for (size_t i = 0; i < size; i++)
        {
            destination[i] = source[i];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::computeVariancesFromCov(const data_management::NumericTable & covariance,
                                                                                   data_management::NumericTable & variances)
{
    size_t nFeatures = covariance.getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, sizeof(algorithmFPType));
    ReadRows<algorithmFPType, cpu> covarianceBlock(const_cast<data_management::NumericTable &>(covariance), 0, nFeatures);
    DAAL_CHECK_BLOCK_STATUS(covarianceBlock);
    const algorithmFPType * covarianceArray = covarianceBlock.get();

    WriteRows<algorithmFPType, cpu> variancesBlock(variances, 0, 1);
    DAAL_CHECK_MALLOC(variancesBlock.get());
    algorithmFPType * variancesArray = variancesBlock.get();

    for (size_t i = 0ul; i < nFeatures; ++i)
    {
        variancesArray[i] = covarianceArray[i * nFeatures + i];
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::correlationFromCovarianceTable(NumericTable & covariance) const
{
    size_t nFeatures = covariance.getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, sizeof(algorithmFPType));
    /* Calculate resulting correlation matrix */
    TArray<algorithmFPType, cpu> diagInvSqrtsArray(nFeatures);
    DAAL_CHECK_MALLOC(diagInvSqrtsArray.get());

    WriteRows<algorithmFPType, cpu> covarianceBlock(covariance, 0, nFeatures);
    DAAL_CHECK_BLOCK_STATUS(covarianceBlock);
    algorithmFPType * covarianceArray = covarianceBlock.get();

    algorithmFPType * diagInvSqrts = diagInvSqrtsArray.get();
    for (size_t i = 0ul; i < nFeatures; ++i)
    {
        diagInvSqrts[i] = 1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(covarianceArray[i * nFeatures + i]);
    }

    for (size_t i = 0ul; i < nFeatures; ++i)
    {
        for (size_t j = 0; j < i; j++)
        {
            covarianceArray[i * nFeatures + j] *= diagInvSqrts[i] * diagInvSqrts[j];
        }
        covarianceArray[i * nFeatures + i] = 1.0; //diagonal element
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0ul; i < nFeatures; ++i)
    {
        for (size_t j = 0; j < i; j++)
        {
            covarianceArray[j * nFeatures + i] = covarianceArray[i * nFeatures + j];
        }
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::copyVarianceFromCovarianceTable(NumericTable & source, NumericTable & dest) const
{
    size_t nElements = dest.getNumberOfColumns();
    size_t nFeatures = source.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> covarianceBlock(source, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(covarianceBlock);
    const algorithmFPType * covarianceArray = covarianceBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> destBlock(dest, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(destBlock);
    algorithmFPType * destData = destBlock.get();

    for (size_t id = 0; id < nElements; ++id)
    {
        destData[id] = covarianceArray[id * (nFeatures + 1)];
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::computeSingularValues(const data_management::NumericTable & eigenvalues,
                                                                                 data_management::NumericTable & singular_values, size_t nRows)
{
    typedef daal::internal::MathInst<algorithmFPType, cpu> Math;
    const size_t nComponents = eigenvalues.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> eigenValuesBlock(const_cast<data_management::NumericTable &>(eigenvalues), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(eigenValuesBlock);
    const algorithmFPType * const eigenValuesArray = eigenValuesBlock.get();
    WriteRows<algorithmFPType, cpu> singularValuesBlock(singular_values, 0, 1);
    DAAL_CHECK_MALLOC(singularValuesBlock.get());
    algorithmFPType * singularValuesArray = singularValuesBlock.get();
    const algorithmFPType factor          = nRows - 1;
    for (size_t i = 0ul; i < nComponents; ++i)
    {
        singularValuesArray[i] = factor * eigenValuesArray[i];
    }
    Math::vSqrt(nComponents, singularValuesArray, singularValuesArray);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::computeCorrelationEigenvalues(const data_management::NumericTable & correlation,
                                                                                         data_management::NumericTable & eigenvectors,
                                                                                         data_management::NumericTable & eigenvalues)
{
    using data_management::BlockDescriptor;

    const size_t nFeatures   = correlation.getNumberOfColumns();
    const size_t nComponents = eigenvalues.getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> correlationBlock(const_cast<data_management::NumericTable &>(correlation), 0, nFeatures);
    DAAL_CHECK_BLOCK_STATUS(correlationBlock);
    const algorithmFPType * correlationArray = correlationBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, nFeatures);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures * nFeatures, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> fullEigenvectors(nFeatures * nFeatures);
    DAAL_CHECK_MALLOC(fullEigenvectors.get());
    algorithmFPType * fullEigenvectorsArray = fullEigenvectors.get();

    TArray<algorithmFPType, cpu> fullEigenvalues(nFeatures);
    DAAL_CHECK_MALLOC(fullEigenvalues.get());
    algorithmFPType * fullEigenvaluesArray = fullEigenvalues.get();

    copyArray(nFeatures * nFeatures, correlationArray, fullEigenvectorsArray);

    services::Status s = computeEigenvectorsInplace(nFeatures, fullEigenvectorsArray, fullEigenvaluesArray);
    DAAL_CHECK_STATUS_VAR(s);

    s = sortEigenvectorsDescending(nFeatures, fullEigenvectorsArray, fullEigenvaluesArray);
    DAAL_CHECK_STATUS_VAR(s);

    WriteOnlyRows<algorithmFPType, cpu> eigenvectorsBlock(eigenvectors, 0, nComponents);
    DAAL_CHECK_BLOCK_STATUS(eigenvectorsBlock);
    algorithmFPType * eigenvectorsArray = eigenvectorsBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> eigenvaluesBlock(eigenvalues, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(eigenvaluesBlock);
    algorithmFPType * eigenvaluesArray = eigenvaluesBlock.get();

    copyArray(nFeatures * nComponents, fullEigenvectorsArray, eigenvectorsArray);
    copyArray(nComponents, fullEigenvaluesArray, eigenvaluesArray);

    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::computeEigenvectorsInplace(size_t nFeatures, algorithmFPType * eigenvectors,
                                                                                      algorithmFPType * eigenvalues)
{
    char jobz = 'V';
    char uplo = 'U';

    DAAL_INT lwork  = 2 * nFeatures * nFeatures + 6 * nFeatures + 1;
    DAAL_INT liwork = 5 * nFeatures + 3;
    DAAL_INT info;

    TArray<algorithmFPType, cpu> work(lwork);
    TArray<DAAL_INT, cpu> iwork(liwork);
    DAAL_CHECK_MALLOC(work.get() && iwork.get());

    LapackInst<algorithmFPType, cpu>::xsyevd(&jobz, &uplo, (DAAL_INT *)(&nFeatures), eigenvectors, (DAAL_INT *)(&nFeatures), eigenvalues, work.get(),
                                             &lwork, iwork.get(), &liwork, &info);
    if (info != 0) return services::Status(services::ErrorPCAFailedToComputeCorrelationEigenvalues);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationBase<algorithmFPType, cpu>::sortEigenvectorsDescending(size_t nFeatures, algorithmFPType * eigenvectors,
                                                                                      algorithmFPType * eigenvalues)
{
    for (size_t i = 0; i < nFeatures / 2; i++)
    {
        const algorithmFPType tmp      = eigenvalues[i];
        eigenvalues[i]                 = eigenvalues[nFeatures - 1 - i];
        eigenvalues[nFeatures - 1 - i] = tmp;
    }

    TArray<algorithmFPType, cpu> eigenvectorTmp(nFeatures);
    DAAL_CHECK_MALLOC(eigenvectorTmp.get());
    for (size_t i = 0; i < nFeatures / 2; i++)
    {
        copyArray(nFeatures, eigenvectors + i * nFeatures, eigenvectorTmp.get());
        copyArray(nFeatures, eigenvectors + nFeatures * (nFeatures - 1 - i), eigenvectors + i * nFeatures);
        copyArray(nFeatures, eigenvectorTmp.get(), eigenvectors + nFeatures * (nFeatures - 1 - i));
    }
    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
