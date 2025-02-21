/* file: adaboost_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Ada Boost algorithm container -- a class that contains
//  Freund Ada Boost kernels for supported architectures.
//--
*/

#include "algorithms/boosting/adaboost_training_batch.h"
#include "src/algorithms/adaboost/adaboost_train_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(adaboost::training::BatchContainer, batch, DAAL_FPTYPE, adaboost::training::defaultDense)
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(adaboost::training::BatchContainer, batch, DAAL_FPTYPE, adaboost::training::sammeR)

namespace adaboost
{
namespace training
{
namespace interface2
{
template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
}

template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const Batch & other) : classifier::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

#define INSTANTIATE_CONSTRUCTORS(algorithmFPType, method)   \
    template Batch<algorithmFPType, method>::Batch(size_t); \
    template Batch<algorithmFPType, method>::Batch(const Batch &);

INSTANTIATE_CONSTRUCTORS(DAAL_FPTYPE, adaboost::training::defaultDense);
INSTANTIATE_CONSTRUCTORS(DAAL_FPTYPE, adaboost::training::sammeR);

} // namespace interface2
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal
