/* file: mt2203_batch_container.h */
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
//  Implementation of mt2203 calculation algorithm container.
//--
*/

#ifndef __MT2203_BATCH_CONTAINER_H__
#define __MT2203_BATCH_CONTAINER_H__

#include "src/algorithms/engines/mt2203/mt2203.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/engines/mt2203/mt2203_kernel.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt2203
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__MT2203__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the mt2203 engine.
 *        This class is associated with the \ref mt2203::internal::Batch "mt2203::Batch" class
 *        and supports the method of mt2203 engine computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of mt2203 engine, double or float
 * \tparam method           Computation method of the engine, mt2203::Method
 * \tparam cpu              Version of the cpu-specific implementation of the engine, CpuType
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the mt2203 engine with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes the result of the mt2203 engine in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::Mt2203Kernel, algorithmFPType, method);
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

    __DAAL_CALL_KERNEL(env, internal::Mt2203Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, resultTable);
}

} // namespace internal
} // namespace mt2203
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
