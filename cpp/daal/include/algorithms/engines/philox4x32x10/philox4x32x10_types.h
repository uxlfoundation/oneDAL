/* file: philox4x32x10_types.h */
/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright contributors to the oneDAL project
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
//  Implementation of the Philox4x32-10 engine: a counter-based pseudorandom number generator (PRNG) 
//  that uses 4x32-bit keys and performs 10 rounds of mixing to produce high-quality randomness.
//--
*/

#ifndef __PHILOX4X32X10_TYPES_H__
#define __PHILOX4X32X10_TYPES_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
/**
 * @defgroup engines_philox4x32x10 philox4x32x10 Engine
 * \copydoc daal::algorithms::engines::philox4x32x10
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for philox4x32x10 engine
 */
namespace philox4x32x10
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__philox4x32x10__METHOD"></a>
 * Available methods to compute philox4x32x10 engine
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

} // namespace philox4x32x10
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
