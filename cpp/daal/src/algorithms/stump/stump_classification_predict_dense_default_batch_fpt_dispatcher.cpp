/* file: stump_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Decision Stump prediction algorithm container --
//  a class that contains Fast Decision Stump kernels for supported architectures.
//--
*/

#include "src/algorithms/stump/stump_classification_predict_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(stump::classification::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      stump::classification::prediction::defaultDense)

namespace stump
{
namespace classification
{
namespace prediction
{
namespace interface1
{
template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
}

template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const Batch & other) : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

template DAAL_EXPORT Batch<DAAL_FPTYPE, stump::classification::prediction::defaultDense>::Batch(size_t nClasses);
template DAAL_EXPORT Batch<DAAL_FPTYPE, stump::classification::prediction::defaultDense>::Batch(const Batch & other);

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
