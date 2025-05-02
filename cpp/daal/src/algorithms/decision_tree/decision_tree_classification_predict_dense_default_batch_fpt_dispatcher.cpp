/* file: decision_tree_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Decision tree algorithm container - a class that contains fast Decision tree prediction kernels for supported
//  architectures.
//--
*/

#include "src/algorithms/decision_tree/decision_tree_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(decision_tree::classification::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      decision_tree::classification::prediction::defaultDense)
namespace decision_tree
{
namespace classification
{
namespace prediction
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, decision_tree::classification::prediction::defaultDense>::Batch(size_t nClasses)
    : classifier::prediction::Batch(), input(), parameter(nClasses)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, decision_tree::classification::prediction::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::prediction::Batch(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface2
} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
