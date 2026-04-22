/* file: mt19937_types.h */
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
//  Implementation of mt19937 engine.
//--
*/

#ifndef __MT19937_TYPES_H__
#define __MT19937_TYPES_H__

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
 * @defgroup engines_mt19937 Mt19937 Engine
 * \copydoc daal::algorithms::engines::mt19937
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mt19937 engine
 */
namespace mt19937
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MT19937__METHOD"></a>
 * Available methods to compute mt19937 engine
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

} // namespace mt19937
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
