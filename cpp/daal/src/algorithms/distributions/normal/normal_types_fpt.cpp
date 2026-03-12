/* file: normal_types_fpt.cpp */
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
//  Implementation of normal distribution algorithm and types methods.
//--

#include "src/algorithms/distributions/normal/normal_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace normal
{
namespace internal
{
/**
  * Check the correctness of the %Parameter object
  */
template <typename algorithmFPType>
services::Status Parameter<algorithmFPType>::check() const
{
    DAAL_CHECK_EX(sigma > 0, services::ErrorIncorrectParameter, services::ParameterName, sigmaStr());
    return services::Status();
}

template services::Status Parameter<DAAL_FPTYPE>::check() const;

} // namespace internal
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal
