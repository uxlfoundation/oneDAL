/* file: kernel_function_linear_csr_fast_batch_fpt_dispatcher.cpp */
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
//  Implementation of linear kernel function container for CSR input data.
//--
*/

#include "algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/kernel_function/kernel_function_linear_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kernel_function::linear::BatchContainer, batch, DAAL_FPTYPE, kernel_function::linear::fastCSR)
namespace kernel_function
{
namespace linear
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, kernel_function::linear::fastCSR>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, kernel_function::linear::fastCSR>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : KernelIface(other), parameter(other.parameter), input(other.input)
{
    initialize();
}
} // namespace interface1
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
