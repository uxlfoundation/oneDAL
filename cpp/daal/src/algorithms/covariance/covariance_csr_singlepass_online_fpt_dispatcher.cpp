/* file: covariance_csr_singlepass_online_fpt_dispatcher.cpp */
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
//  Instantiation of batch covariance calculation algorithm container.
//--
*/

#include "src/algorithms/covariance/covariance_container.h"
#include "src/algorithms/algorithm_dispatch_container_common.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(covariance::internal::OnlineContainer, online, DAAL_FPTYPE, covariance::singlePassCSR)
namespace covariance
{
namespace interface1
{
using OnlineType = Online<DAAL_FPTYPE, covariance::singlePassCSR>;

template <>
void OnlineType::initialize()
{
    this->_ac = new __DAAL_ALGORITHM_CONTAINER(online, internal::OnlineContainer, DAAL_FPTYPE, covariance::singlePassCSR)(&_env);
}

template <>
DAAL_EXPORT OnlineType::Online()
{
    initialize();
}

template <>
DAAL_EXPORT OnlineType::Online(const OnlineType & other) : OnlineImpl(other)
{
    initialize();
}

} // namespace interface1
} // namespace covariance
} // namespace algorithms
} // namespace daal
