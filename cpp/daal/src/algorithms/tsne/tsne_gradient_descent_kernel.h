/* file: tsne_gradient_descent_kernel.h */
/*******************************************************************************
* Copyright 2022 Intel Corporation
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
//  Declaration of template function that calculate tSNE.
//--
*/

#ifndef __TSNE_GRADIENT_DESCENT_KERNEL_H__
#define __TSNE_GRADIENT_DESCENT_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "services/daal_defines.h"
#include "services/env_detect.h"
#include "src/externals/service_math.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace internal
{

template <typename IdxType, typename DataType, daal::internal::CpuType cpu>
services::Status tsneGradientDescentImpl(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                         const NumericTablePtr paramTable, const NumericTablePtr resultTable);

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif // __TSNE_GRADIENT_DESCENT_KERNEL_H__
