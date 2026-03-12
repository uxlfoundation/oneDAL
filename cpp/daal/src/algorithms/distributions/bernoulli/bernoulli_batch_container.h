/* file: bernoulli_batch_container.h */
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
//  Implementation of bernoulli algorithm container.
//--
*/

#ifndef __BERNOULLI_BATCH_CONTAINER_H__
#define __BERNOULLI_BATCH_CONTAINER_H__

#include "src/algorithms/distributions/bernoulli/bernoulli.h"
#include "src/algorithms/distributions/bernoulli/bernoulli_kernel.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace bernoulli
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BernoulliKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    bernoulli::Parameter<algorithmFPType> * parameter = static_cast<bernoulli::Parameter<algorithmFPType> *>(_par);
    ;
    daal::services::Environment::env & env = *_env;

    distributions::Result * result = static_cast<distributions::Result *>(_res);

    result->set(distributions::randomNumbers, static_cast<const distributions::Input *>(_in)->get(distributions::tableToFill));
    NumericTable * resultTable = result->get(distributions::randomNumbers).get();

    __DAAL_CALL_KERNEL(env, internal::BernoulliKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, parameter->p, *parameter->engine,
                       resultTable);
}
} // namespace internal
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
