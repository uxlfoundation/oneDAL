/* file: naivebayes_predict_csr_fast_fpt_dispatcher.cpp */
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
//  Implementation of K-means algorithm container -- a class that contains
//  Lloyd K-means kernels for supported architectures.
//--
*/

#include "src/algorithms/naivebayes/naivebayes_predict_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(multinomial_naive_bayes::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      multinomial_naive_bayes::prediction::fastCSR)
namespace multinomial_naive_bayes
{
namespace prediction
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, multinomial_naive_bayes::prediction::fastCSR>::Batch(size_t nClasses) : parameter(nClasses)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, multinomial_naive_bayes::prediction::fastCSR>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::prediction::Batch(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface2
} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
