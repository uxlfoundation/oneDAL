/* file: bernoulli_dense_default_batch_fpt_dispatcher.cpp */
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

//++
//  Implementation of bernoulli algorithm container.
//--

#include "src/algorithms/distributions/bernoulli/bernoulli_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(distributions::bernoulli::BatchContainer, batch, DAAL_FPTYPE, distributions::bernoulli::defaultDense)

namespace distributions
{
namespace bernoulli
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, distributions::bernoulli::defaultDense>::Batch(DAAL_FPTYPE p) : parameter(p)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, distributions::bernoulli::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : super(other), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal
