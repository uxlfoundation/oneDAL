/* file: lbfgs_batch_container.h */
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
//  Implementation of LBFGS algorithm container.
//--
*/

#include "algorithms/optimization_solver/lbfgs/lbfgs_batch.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_base.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_dense_default_kernel.h"
#include "src/services/service_algo_utils.h"
#include "src/algorithms/engines/engine_factory.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the LBFGS algorithm.
 *        This class is associated with daal::algorithms::optimization_solver::lbfgs::Batch class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the LBFGS algorithm, double or float
 * \tparam method           Stochastic gradient descent computation method, daal::algorithms::optimization_solver::lbfgs::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the limited-memory BFGS algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the limited-memory BFGS algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LBFGSKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input         = static_cast<Input *>(_in);
    Result * result       = static_cast<Result *>(_res);
    Parameter * parameter = static_cast<Parameter *>(_par);

    daal::services::Environment::env & env = *_env;

    NumericTable * correctionPairsInput      = input->get(lbfgs::correctionPairs).get();
    NumericTable * correctionIndicesInput    = input->get(lbfgs::correctionIndices).get();
    NumericTable * inputArgument             = input->get(iterative_solver::inputArgument).get();
    NumericTable * averageArgLIterInput      = input->get(averageArgumentLIterations).get();
    OptionalArgument * optionalArgumentInput = input->get(iterative_solver::optionalArgument).get();

    NumericTable * correctionPairsResult      = result->get(lbfgs::correctionPairs).get();
    NumericTable * correctionIndicesResult    = result->get(correctionIndices).get();
    NumericTable * minimum                    = result->get(iterative_solver::minimum).get();
    NumericTable * nIterations                = result->get(iterative_solver::nIterations).get();
    NumericTable * averageArgLIterResult      = result->get(averageArgumentLIterations).get();
    OptionalArgument * optionalArgumentResult = result->get(iterative_solver::optionalResult).get();
    engines::EnginePtr enginePtr = engines::createEngine(parameter->engine);
    __DAAL_CALL_KERNEL(env, internal::LBFGSKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), correctionPairsInput, correctionIndicesInput, inputArgument, averageArgLIterInput,
                       optionalArgumentInput, correctionPairsResult, correctionIndicesResult, minimum, nIterations, averageArgLIterResult,
                       optionalArgumentResult, parameter, *enginePtr);
}

} // namespace internal
} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
