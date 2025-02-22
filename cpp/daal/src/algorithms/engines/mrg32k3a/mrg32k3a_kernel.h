/* file: mrg32k3a_kernel.h */
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

//++
//  Declaration of a template function for calculating values using the MRG32k3a generator.
//--

#ifndef __MRG32K3A_KERNEL_H__
#define __MRG32K3A_KERNEL_H__

#include "algorithms/engines/mrg32k3a/mrg32k3a.h"
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
namespace mrg32k3a
{
namespace internal
{
/**
 *  \brief Kernel for mrg32k3a calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class mrg32k3aKernel : public Kernel
{
public:
    Status compute(NumericTable * resultTable);
};

} // namespace internal
} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
