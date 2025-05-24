/* file: mt2203_kernel.h */
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
//  Declaration of template function that calculate mt2203s.
//--

#ifndef __MT2203_KERNEL_H__
#define __MT2203_KERNEL_H__

#include "algorithms/engines/mt2203/mt2203.h"
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
namespace mt2203
{
namespace internal
{
/**
 *  \brief Kernel for mt2203 calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class Mt2203Kernel : public Kernel
{
public:
    Status compute(NumericTable * resultTable);
};

} // namespace internal
} // namespace mt2203
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
