/* file: df_regression_train_container.h */
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
//  Implementation of decision forest container.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_CONTAINER_H__
#define __DF_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_training_batch.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for decision forest regression
 *        model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for decision forest model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision forest model-based training in the batch processing mode
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

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const NumericTable * const x = input->get(data).get();
    const NumericTable * const y = input->get(dependentVariable).get();
    const NumericTable * const w = input->get(weights).get();

    decision_forest::regression::Model * m = result->get(model).get();

    const decision_forest::regression::training::Parameter * par = static_cast<decision_forest::regression::training::Parameter *>(_par);
    daal::services::Environment::env & env                       = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), x, y, w, *m, *result, *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                                          = static_cast<Result *>(_res);
    decision_forest::regression::Model * m                   = result->get(model).get();
    decision_forest::regression::internal::ModelImpl * pImpl = dynamic_cast<decision_forest::regression::internal::ModelImpl *>(m);
    if (pImpl == NULL)
    {
        return services::Status(services::ErrorNullResult);
    }
    pImpl->clear();
    return services::Status();
}
} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
