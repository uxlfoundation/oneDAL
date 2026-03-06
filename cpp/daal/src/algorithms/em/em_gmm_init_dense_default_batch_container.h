/* file: em_gmm_init_dense_default_batch_container.h */
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
//  Implementation of EM calculation algorithm container.
//--
*/

#ifndef __EM_GMM_INIT_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __EM_GMM_INIT_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "src/algorithms/em/em_gmm_init_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{
namespace internal
{
/**
 *  \brief Initialize list of em default init kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__BATCHCONTAINER"></a>
 * \brief Provides methods to compute initial values for the EM for GMM algorithm.
 *        The class is associated with the daal::algorithms::em_gmm::init::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial values for the EM for GMM algorithm, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the EM for GMM initialization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes initial values for the EM for GMM algorithm in the batch processing mode
     */
    virtual services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::EMInitKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input = static_cast<Input *>(_in);
    Result * pRes = static_cast<Result *>(_res);

    Parameter * emPar = static_cast<Parameter *>(_par);

    NumericTable * inputData                            = input->get(data).get();
    NumericTable * inputWeights                         = pRes->get(weights).get();
    NumericTable * inputMeans                           = pRes->get(means).get();
    data_management::DataCollectionPtr inputCovariances = pRes->get(covariances);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::EMInitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputData, *inputWeights, *inputMeans,
                       inputCovariances, *emPar, *emPar->engine);
}

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
