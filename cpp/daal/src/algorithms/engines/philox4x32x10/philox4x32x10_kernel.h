/* file: philox4x32x10_kernel.h */
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

//++
//  Declaration of a template function for generating values using the Philox4x32-10 engine.
//--

#ifndef __philox4x32x10_KERNEL_H__
#define __philox4x32x10_KERNEL_H__

#include "algorithms/engines/philox4x32x10/philox4x32x10.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace philox4x32x10
{
namespace internal
{
/**
 *  \brief Kernel for philox4x32x10 calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class philox4x32x10Kernel : public Kernel
{
public:
    Status compute(NumericTable * resultTable);
};

} // namespace internal
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
