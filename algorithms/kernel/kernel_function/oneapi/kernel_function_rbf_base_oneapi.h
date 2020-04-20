/* file: kernel_function_rbf_base_oneapi.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Declaration of template structs that calculate SVM RBF Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_BASE_ONEAPI_H__
#define __KERNEL_FUNCTION_RBF_BASE_ONEAPI_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "algorithms/kernel/kernel.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::data_management;

template <Method method, typename algorithmFPType>
class KernelImplRBFOneAPI : public Kernel
{
public:
    services::Status compute(NumericTable * a1, NumericTable * a2, NumericTable * r, const ParameterBase * par);
};

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
