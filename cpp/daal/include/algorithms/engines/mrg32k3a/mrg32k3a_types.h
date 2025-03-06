/* file: mrg32k3a_types.h */
/*******************************************************************************
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
//  Implementation of the MRG32k3a engine: a 32-bit combined multiple recursive generator
//  with two components of order 3, optimized for batch processing.
//--
*/

#ifndef __MRG32K3A_TYPES_H__
#define __MRG32K3A_TYPES_H__

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
 * @defgroup engines_mrg32k3a mrg32k3a Engine
 * \copydoc daal::algorithms::engines::mrg32k3a
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mrg32k3a engine
 */
namespace mrg32k3a
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__mrg32k3a__METHOD"></a>
 * Available methods to compute mrg32k3a engine
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

} // namespace mrg32k3a
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
