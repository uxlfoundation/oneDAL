/* file: kernel_function_factory.cpp */
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
//  Implementation of kernel function factory.
//--
*/

#include "algorithms/kernel_function/kernel_function.h"
#include "src/algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/kernel_function/kernel_function_rbf.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
template <typename algorithmFPType>
DAAL_EXPORT KernelIfacePtr createKernelFunction(KernelFunctionType type)
{
    switch (type)
    {
    case linearKernel: return KernelIfacePtr(new linear::internal::Batch<algorithmFPType>());
    case rbfKernel: return KernelIfacePtr(new rbf::internal::Batch<algorithmFPType>());
    default: return KernelIfacePtr();
    }
}

// Explicit instantiations
template DAAL_EXPORT KernelIfacePtr createKernelFunction<float>(KernelFunctionType type);
template DAAL_EXPORT KernelIfacePtr createKernelFunction<double>(KernelFunctionType type);

} // namespace interface1
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
