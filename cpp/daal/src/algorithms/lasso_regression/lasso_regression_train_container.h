/* file: lasso_regression_train_container.h */
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
//  Implementation of lasso regression container.
//--
*/

#ifndef __LASSO_REGRESSION_TRAIN_CONTAINER_H__
#define __LASSO_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/lasso_regression/lasso_regression_training_types.h"
#include "algorithms/lasso_regression/lasso_regression_training_batch.h"
#include "src/algorithms/lasso_regression/lasso_regression_train_kernel.h"
#include "src/algorithms/lasso_regression/lasso_regression_model_impl.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace training
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TrainBatchKernel, algorithmFPType, method);
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
    auto x          = input->get(data);
    auto y          = input->get(dependentVariables);
    NumericTablePtr gramMatrix(input->get(training::gramMatrix));
    lasso_regression::Model * m                       = result->get(model).get();
    const lasso_regression::training::Parameter * par = static_cast<lasso_regression::training::Parameter *>(_par);
    daal::services::Environment::env & env            = *_env;
    services::SharedPtr<daal::algorithms::optimization_solver::mse::Batch<algorithmFPType> > objFunc(
        new daal::algorithms::optimization_solver::mse::Batch<algorithmFPType>(x->getNumberOfRows()));
    __DAAL_CALL_KERNEL(env, internal::TrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::getHostApp(*input), x, y, *m, *result, *par, objFunc);
}

} // namespace training
} // namespace lasso_regression
} // namespace algorithms
} // namespace daal
#endif
