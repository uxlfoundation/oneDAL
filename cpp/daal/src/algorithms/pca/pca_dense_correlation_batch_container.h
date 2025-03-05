/* file: pca_dense_correlation_batch_container.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BATCH_CONTAINER_H__
#define __PCA_DENSE_CORRELATION_BATCH_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_batch.h"
#include "src/algorithms/pca/pca_dense_correlation_batch_kernel.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, correlationDense, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCACorrelationKernel, batch, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, correlationDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status BatchContainer<algorithmFPType, correlationDense, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);
    interface3::BatchParameter<algorithmFPType, correlationDense> * parameter =
        static_cast<interface3::BatchParameter<algorithmFPType, correlationDense> *>(_par);
    services::Environment::env & env = *_env;

    data_management::NumericTablePtr data         = input->get(pca::data);
    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);
    data_management::NumericTablePtr means        = result->get(pca::means);
    data_management::NumericTablePtr variances    = result->get(pca::variances);

    auto covarianceAlgorithm = parameter->covariance;
    covarianceAlgorithm->input.set(covariance::data, data);

    if (parameter->resultsToCompute & mean)
    {
        covarianceAlgorithm->getResult()->set(covariance::mean, means);
    }

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(batch, algorithmFPType), compute, input->isCorrelation(),
                       parameter->isDeterministic, *data, covarianceAlgorithm.get(), parameter->resultsToCompute, *eigenvectors, *eigenvalues, *means,
                       *variances);
}

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
