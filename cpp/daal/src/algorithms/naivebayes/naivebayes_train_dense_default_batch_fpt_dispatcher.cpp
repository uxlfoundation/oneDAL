/* file: naivebayes_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Naive Bayes algorithm container -- a class that contains
//  Naive Bayes kernels for supported architectures.
//--
*/

#include "src/algorithms/naivebayes/naivebayes_train_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(multinomial_naive_bayes::training::BatchContainer, batch, DAAL_FPTYPE,
                                      multinomial_naive_bayes::training::defaultDense)
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch(size_t nClasses) : parameter(nClasses)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::training::Batch(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface2
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
