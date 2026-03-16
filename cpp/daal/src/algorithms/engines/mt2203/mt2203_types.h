/* file: mt2203_types.h */
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
//  Implementation of mt2203 engine.
//--
*/

#ifndef __MT2203_TYPES_H__
#define __MT2203_TYPES_H__

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
 * @defgroup engines_mt2203 Mt2203 Engine
 * \copydoc daal::algorithms::engines::mt2203
 * @ingroup engines
 * @{
 */
/**
 * \brief Contains classes for mt2203 engine
 */
namespace mt2203
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__MT2203__METHOD"></a>
 * Available methods to compute mt2203 engine
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

} // namespace mt2203
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
