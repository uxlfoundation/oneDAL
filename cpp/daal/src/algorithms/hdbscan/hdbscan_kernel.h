/* file: hdbscan_kernel.h */
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

#ifndef __HDBSCAN_KERNEL_H__
#define __HDBSCAN_KERNEL_H__

#include "algorithms/hdbscan/hdbscan_types.h"
#include "src/algorithms/kernel.h"
#include "src/services/cpu_type.h"
#include "data_management/data/numeric_table.h"

using daal::data_management::NumericTable;
using daal::internal::CpuType;

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

template <typename algorithmFPType, Method method, CpuType cpu>
class HDBSCANBatchKernel : public Kernel
{
public:
    services::Status compute(const NumericTable * ntData, NumericTable * ntAssignments, NumericTable * ntNClusters, size_t minClusterSize,
                             size_t minSamples);
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_KERNEL_H__
