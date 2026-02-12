/* file: precomputed_types.h */
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
//  Implementation of the objective function with precomputed characteristics
//  interface.
//--
*/

#ifndef __PRECOMPUTED_TYPES_H__
#define __PRECOMPUTED_TYPES_H__

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup precomputed Objective function with precomputed characteristics
 * \copydoc daal::algorithms::optimization_solver::precomputed
 * @ingroup objective_function
 * @{
 */
/**
 * \brief Contains classes for the Objective function with precomputed characteristics
 */
namespace precomputed
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__METHOD"></a>
 * Available methods for computing results of Objective function with precomputed characteristics
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};
} // namespace precomputed
/** @} */
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
