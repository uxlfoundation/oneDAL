/* file: algorithm_kernel.h */
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
//  Implementation of base classes defining algorithm kernel.
//--
*/

#ifndef __ALGORITHM_KERNEL_H__
#define __ALGORITHM_KERNEL_H__

#include "services/daal_memory.h"
#include "services/error_handling.h"
#include "services/env_detect.h"
#include "algorithms/algorithm_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL"></a>
 * \brief %Base class to represent algorithm implementation
 */
class Kernel
{
public:
    Kernel() {};

    virtual ~Kernel() {}
};

/** @} */
} // namespace interface1
using interface1::Kernel;

} // namespace algorithms
} // namespace daal
#endif
