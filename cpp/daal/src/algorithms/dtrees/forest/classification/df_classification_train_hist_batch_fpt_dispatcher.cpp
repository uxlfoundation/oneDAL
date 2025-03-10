/* file: df_classification_train_hist_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of decision forest container for the hist method.
//--
*/

#include "src/algorithms/dtrees/forest/classification/df_classification_train_container.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_SAFE(decision_forest::classification::training::BatchContainer, batch, DAAL_FPTYPE,
                                           decision_forest::classification::training::hist)
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace interface3
{
using BatchType = Batch<DAAL_FPTYPE, decision_forest::classification::training::hist>;

template <>
DAAL_EXPORT BatchType::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
    parameter().minObservationsInLeafNode = 1;
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

template <>
DAAL_EXPORT services::Status BatchType::checkComputeParams()
{
    services::Status s = classifier::training::Batch::checkComputeParams();
    if (!s) return s;
    const auto x         = input.get(classifier::training::data);
    const auto nFeatures = x->getNumberOfColumns();
    DAAL_CHECK_EX(parameter().featuresPerNode <= nFeatures, services::ErrorIncorrectParameter, services::ParameterName, featuresPerNodeStr());
    const size_t nSamplesPerTree(parameter().observationsPerTreeFraction * x->getNumberOfRows());
    DAAL_CHECK_EX(nSamplesPerTree > 0, services::ErrorIncorrectParameter, services::ParameterName, observationsPerTreeFractionStr());
    return s;
}
} // namespace interface3
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
