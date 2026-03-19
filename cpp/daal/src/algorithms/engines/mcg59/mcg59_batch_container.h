/* file: mcg59_batch_container.h */
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
//  Implementation of mcg59 calculation algorithm container.
//--
*/

#ifndef __MCG59_BATCH_CONTAINER_H__
#define __MCG59_BATCH_CONTAINER_H__

#include "algorithms/engines/mcg59/mcg59.h"
#include "src/algorithms/engines/mcg59/mcg59_kernel.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mcg59
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__MCG59__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the mcg59 engine.
 *        This class is associated with the \ref mcg59::interface1::Batch "mcg59::Batch" class
 *        and supports the method of mcg59 engine computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of mcg59 engine, double or float
 * \tparam method           Computation method of the engine, mcg59::Method
 * \tparam cpu              Version of the cpu-specific implementation of the engine, CpuType
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the mcg59 engine with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes the result of the mcg59 engine in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::Mcg59Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    daal::services::Environment::env & env = *_env;
    engines::Result * result               = static_cast<engines::Result *>(_res);
    NumericTable * resultTable             = result->get(engines::randomNumbers).get();

    __DAAL_CALL_KERNEL(env, internal::Mcg59Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, resultTable);
}

} // namespace internal
} // namespace mcg59
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
