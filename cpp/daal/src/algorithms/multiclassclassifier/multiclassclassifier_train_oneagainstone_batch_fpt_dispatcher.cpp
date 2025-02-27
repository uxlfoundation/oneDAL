/* file: multiclassclassifier_train_oneagainstone_batch_fpt_dispatcher.cpp */
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
//  Instantiation of Multi-class classifier training algorithm container.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_train.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(multi_class_classifier::training::BatchContainer, batch, DAAL_FPTYPE,
                                      multi_class_classifier::training::oneAgainstOne)
namespace multi_class_classifier
{
namespace training
{
namespace interface2
{

template <typename algorithmFPType, Method method>
DAAL_DEPRECATED Batch<algorithmFPType, method>::Batch() : parameter(0)
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(size_t nClasses) : parameter(nClasses)
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(const Batch<algorithmFPType, method> & other)
    : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
{
    initialize();
}

template class Batch<DAAL_FPTYPE, training::oneAgainstOne>;
} // namespace interface2
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
