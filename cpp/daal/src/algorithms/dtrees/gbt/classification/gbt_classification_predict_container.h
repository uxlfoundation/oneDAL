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
namespace interface2
{
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
    __DAAL_CALL_KERNEL(env, internal::PredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a, m, r, prob, par->nClasses,
                       par->nIterations, predShapContributions, predShapInteractions);
}

} // namespace interface2
} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
