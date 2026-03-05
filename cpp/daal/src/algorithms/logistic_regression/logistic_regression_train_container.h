/* file: logistic_regression_train_container.h */
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
//  Implementation of logistic regression container.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_CONTAINER_H__
#define __LOGISTIC_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/logistic_regression/logistic_regression_training_types.h"
#include "algorithms/logistic_regression/logistic_regression_training_batch.h"
#include "src/algorithms/logistic_regression/logistic_regression_train_kernel.h"
#include "src/algorithms/logistic_regression/logistic_regression_model_impl.h"
#include "algorithms/optimization_solver/lbfgs/lbfgs_batch.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace interface3
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
    classifier::training::Input * input                  = static_cast<classifier::training::Input *>(_in);
    Result * result                                      = static_cast<Result *>(_res);
    auto x                                               = input->get(classifier::training::data);
    auto y                                               = input->get(classifier::training::labels);
    logistic_regression::Model * m                       = result->get(classifier::training::model).get();
    const logistic_regression::training::Parameter * par = static_cast<logistic_regression::training::Parameter *>(_par);
    daal::services::Environment::env & env               = *_env;

    __DAAL_CALL_KERNEL(env, internal::TrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::getHostApp(*input), x, y, *m, *result, *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Result * result                                  = static_cast<Result *>(_res);
    logistic_regression::Model * m                   = result->get(classifier::training::model).get();
    logistic_regression::internal::ModelImpl * pImpl = dynamic_cast<logistic_regression::internal::ModelImpl *>(m);

    logistic_regression::training::Parameter * par = static_cast<logistic_regression::training::Parameter *>(_par);
    if (!par->optimizationSolver.get())
    {
        auto solver                             = optimization_solver::lbfgs::Batch<algorithmFPType>::create();
        par->optimizationSolver                 = solver;
        const size_t nIterations                = 1000;
        const algorithmFPType stepLength        = 1.0;
        const algorithmFPType accuracyThreshold = 1e-4;
        solver->parameter.stepLengthSequence    = HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, stepLength);
        solver->parameter.accuracyThreshold     = accuracyThreshold;
        solver->parameter.nIterations           = nIterations;
        classifier::training::Input * input     = static_cast<classifier::training::Input *>(_in);
        solver->parameter.batchSize             = input->get(classifier::training::data)->getNumberOfRows();
    }
    DAAL_CHECK(pImpl, ErrorNullPtr);
    return pImpl->reset(par->interceptFlag);
}

} // namespace interface3
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
