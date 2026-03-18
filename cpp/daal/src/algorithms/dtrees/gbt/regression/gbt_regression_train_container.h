/* file: gbt_regression_train_container.h */
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
//  Implementation of gradient boosted trees container.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_CONTAINER_H__
#define __GBT_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_training_batch.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_train_kernel.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include "src/services/service_algo_utils.h"
#include "src/algorithms/engines/engine_factory.h"
namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{
/**
 *  \brief Initialize list of gradient boosted trees
 *  kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for gradient boosted trees regression
 *        model-based training using algorithmFPType precision arithmetic
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of model-based training in the batch processing mode
     * \return Status of computations
     */
    services::Status compute() override;
    services::Status setupCompute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate gradient boosted trees model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Decision forest algorithm parameters
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const NumericTable * x = input->get(data).get();
    const NumericTable * y = input->get(dependentVariable).get();

    gbt::regression::Model * m = result->get(model).get();

    const Parameter * par                  = static_cast<gbt::regression::training::Parameter *>(_par);
    daal::services::Environment::env & env = *_env;
    engines::EnginePtr enginePtr = engines::createEngine(par->engine);
    daal::algorithms::engines::internal::BatchBaseImpl * engine =
        dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(enginePtr.get());

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), x, y, *m, *result, *par, *engine);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                              = static_cast<Result *>(_res);
    gbt::regression::Model * m                   = result->get(model).get();
    gbt::regression::internal::ModelImpl * pImpl = dynamic_cast<gbt::regression::internal::ModelImpl *>(m);
    DAAL_CHECK(pImpl != nullptr, services::ErrorIncorrectTypeOfModel)
    pImpl->clear();
    return services::Status();
}

} // namespace internal

} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
