/* file: bernoulli_types.h */
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
//  Implementation of bernoulli distribution.
//--
*/

#ifndef __BERNOULLI__TYPES_H__
#define __BERNOULLI__TYPES_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @defgroup distributions_bernoulli Bernoulli Distribution
 * \copydoc daal::algorithms::distributions::bernoulli
 * @ingroup distributions
 * @{
 */
/**
 * \brief Contains classes for bernoulli distribution
 */
namespace bernoulli
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__METHOD"></a>
 * Available methods to compute bernoulli distribution
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__PARAMETER"></a>
 * \brief Bernoulli distribution parameters
 */
template <typename algorithmFPType>
class DAAL_EXPORT Parameter : public distributions::ParameterBase
{
public:
    /**
     *  Main constructor
     *  \param[in] _p    Success probability of a trial, value from [0.0; 1.0]
     */
    Parameter(algorithmFPType _p) : p(_p) {}

    algorithmFPType p; /*!< Success probability of a trial, value from [0.0; 1.0] */

    /**
     * Check the correctness of the %Parameter object
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace bernoulli
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
