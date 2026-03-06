/* file: uniform_batch_container.h */
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
//  Implementation of uniform algorithm container.
//--
*/

#ifndef __UNIFORM_BATCH_CONTAINER_H__
#define __UNIFORM_BATCH_CONTAINER_H__

#include "algorithms/distributions/uniform/uniform.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__UNIFORM__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the uniform distribution.
 *        This class is associated with the \ref uniform::interface1::Batch "uniform::Batch" class
 *        and supports the method of uniform distribution computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of uniform distribution, double or float
 * \tparam method           Computation method of the distribution, uniform::Method
 * \tparam cpu              Version of the cpu-specific implementation of the distribution, daal::CpuType
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the uniform distribution with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes the result of the uniform distribution in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::UniformKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    uniform::Parameter<algorithmFPType> * parameter = static_cast<uniform::Parameter<algorithmFPType> *>(_par);
    daal::services::Environment::env & env          = *_env;

    distributions::Result * result = static_cast<distributions::Result *>(_res);

    result->set(distributions::randomNumbers, static_cast<const distributions::Input *>(_in)->get(distributions::tableToFill));
    NumericTable * resultTable = result->get(distributions::randomNumbers).get();

    __DAAL_CALL_KERNEL(env, internal::UniformKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *parameter, *parameter->engine,
                       resultTable);
}
} // namespace internal
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
