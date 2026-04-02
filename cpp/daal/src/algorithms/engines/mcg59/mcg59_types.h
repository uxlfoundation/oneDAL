/* file: mcg59_types.h */
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
//  Implementation of mcg59 engine.
//--
*/

#ifndef __MCG59_TYPES_H__
#define __MCG59_TYPES_H__

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
 * @defgroup engines_mcg59 Mcg59 Engine
 * \copydoc daal::algorithms::engines::mcg59
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mcg59 engine
 */
namespace mcg59
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MCG59__METHOD"></a>
 * Available methods to compute mcg59 engine
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

} // namespace mcg59
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
