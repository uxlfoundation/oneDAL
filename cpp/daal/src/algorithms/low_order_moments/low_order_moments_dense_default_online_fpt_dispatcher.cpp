/* file: low_order_moments_dense_default_online_fpt_dispatcher.cpp */
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
//  Instantiation of online low order moments algorithm container.
//--
*/

#include "src/algorithms/low_order_moments/low_order_moments_container.h"
#include "src/algorithms/algorithm_dispatch_container_common.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(low_order_moments::internal::OnlineContainer, online, DAAL_FPTYPE, low_order_moments::defaultDense)
namespace low_order_moments
{
namespace interface1
{
template <>
void Online<DAAL_FPTYPE, low_order_moments::defaultDense>::initialize()
{
    Analysis<online>::_ac = new __DAAL_ALGORITHM_CONTAINER(online, internal::OnlineContainer, DAAL_FPTYPE, low_order_moments::defaultDense)(&_env);
    _in                   = &input;
    _par                  = &parameter;
    _result.reset(new ResultType());
    _partialResult.reset(new PartialResultType());
}
template <>
DAAL_EXPORT Online<DAAL_FPTYPE, low_order_moments::defaultDense>::Online()
{
    initialize();
}

using OnlineType = Online<DAAL_FPTYPE, low_order_moments::defaultDense>;

template <>
DAAL_EXPORT OnlineType::Online(const OnlineType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
