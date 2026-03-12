/* file: distribution_batch.h */
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
//  Implementation of distribution methods.
//--
*/
#ifndef __DISTRIBUTION_BATCH__
#define __DISTRIBUTION_BATCH__

#include "src/algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace interface1
{
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const Input * algInput = static_cast<const Input *>(input);

    set(randomNumbers, algInput->get(tableToFill));
    return services::Status();
}

} // namespace interface1
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
