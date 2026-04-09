/* file: pivoted_qr_kernel.h */
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
//  Declaration of template function that calculate qrs.
//--
*/

#ifndef __PIVOTED_QR_KERNEL_H__
#define __PIVOTED_QR_KERNEL_H__

#include "algorithms/pivoted_qr/pivoted_qr_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
/**
 *  \brief Kernel for QR calculation
 */
template <daal::algorithms::pivoted_qr::Method method, typename algorithmFPType, CpuType cpu>
class PivotedQRKernel : public Kernel
{
public:
    services::Status compute(const NumericTable & dataTable, NumericTable & QTable, NumericTable & RTable, NumericTable & PTable,
                             NumericTable * permutedColumns);
};

} // namespace internal
} // namespace pivoted_qr
} // namespace algorithms
} // namespace daal

#endif
