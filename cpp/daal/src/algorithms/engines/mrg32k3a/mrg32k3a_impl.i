/* file: mrg32k3a_impl.i */
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
//  Implementation of mrg32k3a algorithm.
//--
*/

#ifndef __MRG32K3A_IMPL_I__
#define __MRG32K3A_IMPL_I__

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
template <typename algorithmFPType, Method method, CpuType cpu>
Status mrg32k3aKernel<algorithmFPType, method, cpu>::compute(NumericTable * resultTensor)
{
    return Status();
}

} // namespace internal
} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
