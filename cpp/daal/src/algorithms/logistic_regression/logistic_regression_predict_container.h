/* file: logistic_regression_predict_container.h */
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
//  Implementation of logistic regression algorithm container -- a class
//  that contains fast logistic regression prediction kernels
//  for supported architectures.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict.h"
#include "algorithms/classifier/classifier_model.h"
#include "src/algorithms/logistic_regression/logistic_regression_predict_kernel.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the logistic regression algorithm.
 *        This class is associated with daal::algorithms::logistic_regression::prediction::interface2::Batch class
 *        and supports method to compute logistic regression prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           logistic regression computation method, \ref Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for logistic regression model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of logistic regression model-based prediction
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

    NumericTable * a                  = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    logistic_regression::Model * m    = static_cast<logistic_regression::Model *>(input->get(classifier::prediction::model).get());
    const classifier::Parameter * par = static_cast<classifier::Parameter *>(_par);

    NumericTable * r = ((par->resultsToEvaluate & classifier::computeClassLabels) ? result->get(classifier::prediction::prediction).get() : nullptr);
    NumericTable * prob =
        ((par->resultsToEvaluate & classifier::computeClassProbabilities) ? result->get(classifier::prediction::probabilities).get() : nullptr);
    NumericTable * logProb =
        ((par->resultsToEvaluate & classifier::computeClassLogProbabilities) ? result->get(classifier::prediction::logProbabilities).get() : nullptr);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a, m, par->nClasses, r, prob,
                       logProb);
}

} // namespace internal

} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
