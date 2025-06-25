/* file: cordistance_kernel.h */
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
//  Declaration of kernel class for computation of correlation distance.
//--
*/

#ifndef __CORDISTANCE_KERNEL_H__
#define __CORDISTANCE_KERNEL_H__

#include "algorithms/distance/correlation_distance.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class DistanceKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                             const daal::algorithms::Parameter * par);
};

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal

#endif
