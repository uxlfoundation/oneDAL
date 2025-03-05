/* file: pca_dense_correlation_online_container.h */
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

#ifndef __PCA_DENSE_CORRELATION_ONLINE_CONTAINER_H__
#define __PCA_DENSE_CORRELATION_ONLINE_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_online.h"
#include "src/algorithms/pca/pca_dense_correlation_online_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, correlationDense, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCACorrelationKernel, online, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
OnlineContainer<algorithmFPType, correlationDense, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, correlationDense, cpu>::compute()
{
    Input * input                                                  = static_cast<Input *>(_in);
    OnlineParameter<algorithmFPType, correlationDense> * parameter = static_cast<OnlineParameter<algorithmFPType, correlationDense> *>(_par);
    PartialResult<correlationDense> * partialResult                = static_cast<PartialResult<correlationDense> *>(_pres);
    services::Environment::env & env                               = *_env;

    data_management::NumericTablePtr data = input->get(pca::data);

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(online, algorithmFPType), compute, data, partialResult,
                       parameter);
}

template <typename algorithmFPType, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, correlationDense, cpu>::finalizeCompute()
{
    OnlineParameter<algorithmFPType, correlationDense> * parameter = static_cast<OnlineParameter<algorithmFPType, correlationDense> *>(_par);
    PartialResult<correlationDense> * partialResult                = static_cast<PartialResult<correlationDense> *>(_pres);
    Result * result                                                = static_cast<Result *>(_res);
    services::Environment::env & env                               = *_env;

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(online, algorithmFPType), finalize, partialResult, parameter,
                       *eigenvectors, *eigenvalues);
}

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
