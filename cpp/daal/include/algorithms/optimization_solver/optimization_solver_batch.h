/* file: optimization_solver_batch.h */
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
//  Implementation of Optimization solver interface interface.
//--
*/

#ifndef __OPTIMIZATION_SOLVER_BATCH_H__
#define __OPTIMIZATION_SOLVER_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
* \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
*/
namespace interface1
{
/**
 * @addtogroup optimization_solver
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__BATCHIFACE"></a>
 * \brief Interface for computing the Optimization solver in the %batch processing mode.
 */
class DAAL_EXPORT BatchIface : public daal::algorithms::Analysis<batch>
{
public:
    BatchIface() {}
    virtual ~BatchIface() {}
};
/** @} */
} // namespace interface1
using interface1::BatchIface;

} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
