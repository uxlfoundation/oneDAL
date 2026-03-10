/* file: cholesky_batch_container.h */
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
//  Implementation of cholesky calculation algorithm container.
//--
*/

#include "algorithms/cholesky/cholesky.h"
#include "src/algorithms/cholesky/cholesky_kernel.h"

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Cholesky decomposition algorithm.
 *        This class is associated with daal::algorithms::cholesky::Batch class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Cholesky decomposition algorithm, double or float
 * \tparam method           Cholesky decomposition computation method, \ref daal::algorithms::cholesky::Method
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the Cholesky decomposition algorithm with a specified environment
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the Cholesky decomposition algorithm in the batch processing mode
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::CholeskyKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    daal::algorithms::Parameter * par      = nullptr;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::CholeskyKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input->get(data).get(),
                       result->get(choleskyFactor).get(), par);
}

} // namespace internal
} // namespace cholesky
} // namespace algorithms
} // namespace daal
