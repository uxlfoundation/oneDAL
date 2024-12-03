/* file: philox4x32x10_impl.i */
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

/*
//++
//  Implementation of philox4x32x10 algorithm.
//--
*/

#ifndef __philox4x32x10_IMPL_I__
#define __philox4x32x10_IMPL_I__

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
template <typename algorithmFPType, Method method, CpuType cpu>
Status philox4x32x10Kernel<algorithmFPType, method, cpu>::compute(NumericTable * resultTensor)
{
    return Status();
}

} // namespace internal
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
