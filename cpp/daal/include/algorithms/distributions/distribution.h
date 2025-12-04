/* file: distribution.h */
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
//  Implementation of distributions
//--
*/

#ifndef __DISTRIBUTIONS_H__
#define __DISTRIBUTIONS_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @ingroup distributions
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BATCHBASE"></a>
 * \brief Class representing distributions
 */
class DAAL_EXPORT BatchBase : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::distributions::Input InputType;
    typedef algorithms::distributions::ParameterBase ParameterType;
    typedef algorithms::distributions::Result ResultType;

    InputType input; /*!< Input of the distribution */
    virtual ~BatchBase() {}
};

} // namespace interface1
using interface1::BatchBase;
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal
#endif
