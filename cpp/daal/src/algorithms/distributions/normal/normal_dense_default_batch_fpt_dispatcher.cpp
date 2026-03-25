/* file: normal_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of normal algorithm container.
//--

#include "src/algorithms/distributions/normal/normal_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(distributions::normal::internal::BatchContainer, batch, DAAL_FPTYPE, distributions::normal::defaultDense)
namespace distributions
{
namespace normal
{
namespace internal
{
using BatchType = Batch<DAAL_FPTYPE, distributions::normal::defaultDense>;

template <>
void BatchType::initialize()
{
    Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, internal::BatchContainer, DAAL_FPTYPE, distributions::normal::defaultDense)(&_env);
    _in                  = &input;
    _par                 = &parameter;
    _result.reset(new ResultType());
}

template <>
BatchType::Batch(DAAL_FPTYPE a, DAAL_FPTYPE sigma) : parameter(a, sigma)
{
    initialize();
}

template <>
BatchType::Batch(const BatchType & other) : super(other), parameter(other.parameter)
{
    initialize();
}

} // namespace internal
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal
