/* file: covariance_online_parameter.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "algorithms/covariance/covariance_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
/** Default constructor */
OnlineParameter::OnlineParameter() : Parameter() {}

/**
*  Constructs parameters of the Covariance Online algorithm by copying another parameters of the Covariance Online algorithm
*  \param[in] other    Parameters of the Covariance Online algorithm
*/
OnlineParameter::OnlineParameter(const OnlineParameter & other) = default;

/**
*  Copy-assignment operator for parameters of the Covariance Online algorithm
*  \param[in] other    Parameters of the Covariance Online algorithm
*/
OnlineParameter & OnlineParameter::operator=(const OnlineParameter & other) = default;

/**
 * Check the correctness of the %OnlineParameter object
 */
services::Status OnlineParameter::check() const
{
    return services::Status();
}

} //namespace interface1

} //namespace covariance
} // namespace algorithms
} // namespace daal
