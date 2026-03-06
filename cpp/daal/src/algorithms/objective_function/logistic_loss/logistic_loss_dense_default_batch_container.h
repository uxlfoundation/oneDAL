/* file: logistic_loss_dense_default_batch_container.h */
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
//  Implementation of logloss calculation algorithm container.
//--
*/

#ifndef __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"
#include "src/algorithms/objective_function/logistic_loss/logistic_loss_dense_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Logistic loss objective function.
 *        This class is associated with the Batch class and supports the method of computing
 *        the Logistic loss objective function in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Logistic loss objective function, double or float
 * \tparam method           the Logistic loss objective function computation method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for logistic loss objective function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of logistic loss objective function in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LogLossKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                          = static_cast<Input *>(_in);
    objective_function::Result * result    = static_cast<objective_function::Result *>(_res);
    Parameter * parameter                  = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;
    NumericTable * value                   = nullptr;
    NumericTable * hessian                 = nullptr;
    NumericTable * gradient                = nullptr;
    NumericTable * nonSmoothTermValue      = nullptr;
    NumericTable * proximalProjection      = nullptr;
    NumericTable * lipschitzConstant       = nullptr;

    if (parameter->resultsToCompute & objective_function::value)
    {
        value = result->get(objective_function::valueIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::hessian)
    {
        hessian = result->get(objective_function::hessianIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::gradient)
    {
        gradient = result->get(objective_function::gradientIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::nonSmoothTermValue)
    {
        nonSmoothTermValue = result->get(objective_function::nonSmoothTermValueIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::proximalProjection)
    {
        proximalProjection = result->get(objective_function::proximalProjectionIdx).get();
    }

    if (parameter->resultsToCompute & objective_function::lipschitzConstant)
    {
        lipschitzConstant = result->get(objective_function::lipschitzConstantIdx).get();
    }

    __DAAL_CALL_KERNEL(env, internal::LogLossKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input->get(logistic_loss::data).get(),
                       input->get(logistic_loss::dependentVariables).get(), input->get(logistic_loss::argument).get(), value, hessian, gradient,
                       nonSmoothTermValue, proximalProjection, lipschitzConstant, parameter);
}

} // namespace internal
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
