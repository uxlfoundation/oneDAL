/* file: pca_dense_correlation_batch_impl.i */
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

#ifndef __PCA_DENSE_CORRELATION_BATCH_IMPL_I__
#define __PCA_DENSE_CORRELATION_BATCH_IMPL_I__

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"

#include "src/externals/service_profiler.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<batch, algorithmFPType, cpu>::compute(bool isCorrelation, const data_management::NumericTable & dataTable,
                                                                            covariance::BatchImpl * covarianceAlg,
                                                                            data_management::NumericTable & eigenvectors,
                                                                            data_management::NumericTable & eigenvalues)
{
    if (isCorrelation) return this->computeCorrelationEigenvalues(dataTable, eigenvectors, eigenvalues);
    DAAL_CHECK(covarianceAlg, services::ErrorNullPtr);
    services::Status status;
    covarianceAlg->parameter.outputMatrixType = covariance::correlationMatrix;

    DAAL_CHECK_STATUS(status, covarianceAlg->computeNoThrow());
    return this->computeCorrelationEigenvalues(*covarianceAlg->getResult()->get(covariance::covariance), eigenvectors, eigenvalues);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<batch, algorithmFPType, cpu>::compute(
    bool isCorrelation, bool isDeterministic, const data_management::NumericTable & dataTable, covariance::BatchImpl * covarianceAlg,
    DAAL_UINT64 resultsToCompute, data_management::NumericTable & eigenvectors, data_management::NumericTable & eigenvalues,
    data_management::NumericTable & means, data_management::NumericTable & variances, bool doScale)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    services::Status status;
    if (isCorrelation)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation);
        if (resultsToCompute & mean)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(means, (algorithmFPType)0));
        }

        if (resultsToCompute & variance)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(variances, (algorithmFPType)1));
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.computeEigenvalues);
            DAAL_CHECK_STATUS(status, this->computeCorrelationEigenvalues(dataTable, eigenvectors, eigenvalues));
        }
    }
    else
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.full);

        DAAL_CHECK(covarianceAlg, services::ErrorNullPtr);
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.covariance);
            DAAL_CHECK_STATUS(status, covarianceAlg->computeNoThrow());
        }

        auto pCovarianceTable          = covarianceAlg->getResult()->get(covariance::covariance);
        NumericTable & covarianceTable = *pCovarianceTable;
        if (resultsToCompute & mean)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.copyMeans);
            DAAL_CHECK_STATUS(status, this->copyTable(*covarianceAlg->getResult()->get(covariance::mean), means));
        }

        if (resultsToCompute & variance)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.copyVariances);
            DAAL_CHECK_STATUS(status, this->copyVarianceFromCovarianceTable(covarianceTable, variances));
        }
        if (doScale)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.correlationFromCovariance);
            DAAL_CHECK_STATUS(status, this->correlationFromCovarianceTable(covarianceTable));
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.computeEigenvalues);
            DAAL_CHECK_STATUS(status, this->computeCorrelationEigenvalues(covarianceTable, eigenvectors, eigenvalues));
        }
    }

    if (isDeterministic)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.signFlipEigenvectors);
        DAAL_CHECK_STATUS(status, this->signFlipEigenvectors(eigenvectors));
    }

    return status;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<batch, algorithmFPType, cpu>::compute(
    const data_management::NumericTable & dataTable, covariance::BatchImpl * covarianceAlg, data_management::NumericTable & eigenvectors,
    data_management::NumericTable & eigenvalues, data_management::NumericTable & means, data_management::NumericTable & variances,
    data_management::NumericTable * singular_values, data_management::NumericTable * explained_variances_ratio, const BaseBatchParameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    services::Status status;
    this->compute(parameter->isCorrelation, parameter->isDeterministic, dataTable, covarianceAlg, parameter->resultsToCompute, eigenvectors,
                  eigenvalues, means, variances, parameter->doScale);

    if (singular_values != nullptr)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.computeSingularValues);
        DAAL_CHECK_STATUS(status, this->computeSingularValues(eigenvalues, *singular_values, dataTable.getNumberOfRows()));
    }
    if (explained_variances_ratio != nullptr)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.computeExplainedVariancesRatio);
        DAAL_CHECK_STATUS(status, this->computeExplainedVariancesRatio(eigenvalues, variances, *explained_variances_ratio));
    }
    return status;
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
