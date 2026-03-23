/* file: kernel_function_types.cpp */
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
//  Implementation of kernel function Result.
//--
*/

#include "src/algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/kernel_function/kernel_function_linear_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

} // namespace kernel_function
} // namespace algorithms
} // namespace daal
