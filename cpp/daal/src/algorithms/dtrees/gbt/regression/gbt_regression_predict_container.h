/* file: gbt_regression_predict_container.h */
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
//  Implementation of gradient boosted trees algorithm container -- a class
//  that contains fast gradient boosted trees prediction kernels
//  for supported architectures.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_predict.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for model-based prediction
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     *  Computes the result of model-based prediction
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::PredictKernel, algorithmFPType, method);
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

    NumericTable * a                                   = static_cast<NumericTable *>(input->get(data).get());
    daal::algorithms::gbt::regression::Model * m       = static_cast<daal::algorithms::gbt::regression::Model *>(input->get(model).get());
    NumericTable * r                                   = static_cast<NumericTable *>(result->get(prediction).get());
    const gbt::regression::prediction::Parameter * par = static_cast<gbt::regression::prediction::Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    const bool predShapContributions       = par->resultsToCompute & shapContributions;
    const bool predShapInteractions        = par->resultsToCompute & shapInteractions;
    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a, m, r, par->nIterations,
                       predShapContributions, predShapInteractions);
}

} // namespace internal

} // namespace prediction
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
