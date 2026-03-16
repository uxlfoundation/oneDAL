/* file: gbt_classification_predict_container.h */
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

#include "algorithms/gradient_boosted_trees/gbt_classification_predict.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_predict_kernel.h"
#include "src/services/service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the gradient boosted trees algorithm.
 *        This class is associated with daal::algorithms::gbt::prediction::interface1::Batch class
 *        and supports method to compute gbt prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           gradient boosted trees computation method, \ref Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for gradient boosted trees model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of gbt model-based prediction
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
    Input * input                           = static_cast<Input *>(_in);
    classifier::prediction::Result * result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable * a               = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    gbt::classification::Model * m = static_cast<gbt::classification::Model *>(input->get(classifier::prediction::model).get());

    daal::services::Environment::env & env                 = *_env;
    const gbt::classification::prediction::Parameter * par = static_cast<gbt::classification::prediction::Parameter *>(_par);

    NumericTable * r =
        (par->resultsToEvaluate & classifier::ResultToComputeId::computeClassLabels ? result->get(classifier::prediction::prediction).get() :
                                                                                      nullptr);
    NumericTable * prob = ((par->resultsToEvaluate & classifier::ResultToComputeId::computeClassProbabilities) ?
                               result->get(classifier::prediction::probabilities).get() :
                               nullptr);

    const bool predShapContributions = par->resultsToCompute & shapContributions;
    const bool predShapInteractions  = par->resultsToCompute & shapInteractions;
    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       daal::services::internal::hostApp(*input), a, m, r, prob, par->nClasses, par->nIterations, predShapContributions,
                       predShapInteractions);
}

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
