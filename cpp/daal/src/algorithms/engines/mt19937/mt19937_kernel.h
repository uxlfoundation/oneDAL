/* file: mt19937_kernel.h */
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

//++
//  Declaration of template function that calculate mt19937s.
//--

#ifndef __MT19937_KERNEL_H__
#define __MT19937_KERNEL_H__

#include "src/algorithms/engines/mt19937/mt19937.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt19937
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

/**
 *  \brief Kernel for mt19937 calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class Mt19937Kernel : public Kernel
{
public:
    Status compute(NumericTable * resultTable);
};

} // namespace internal
} // namespace mt19937
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
