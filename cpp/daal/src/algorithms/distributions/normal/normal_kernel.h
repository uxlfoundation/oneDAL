/* file: normal_kernel.h */
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

//++
//  Declaration of template function that calculates normal distribution.
//--

#ifndef __NORMAL_KERNEL_H__
#define __NORMAL_KERNEL_H__

#include "src/algorithms/distributions/normal/normal.h"

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/engines/engine_batch_impl.h"

#include "src/externals/service_rng.h"
#include "src/services/service_unique_ptr.h"
#include "src/data_management/service_numeric_table.h"

using namespace daal::services;
using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace normal
{
namespace internal
{
/**
 *  \brief Kernel for normal calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class NormalKernel : public Kernel
{
public:
    static Status compute(const normal::Parameter<algorithmFPType> * parameter, engines::BatchBase & engine, NumericTable * resultTable);
    static Status compute(const normal::Parameter<algorithmFPType> * parameter, engines::BatchBase & engine, size_t n, algorithmFPType * resultArray);
    static Status compute(const normal::Parameter<algorithmFPType> * parameter, UniquePtr<engines::internal::BatchBaseImpl, cpu> & enginePtr,
                          size_t n, algorithmFPType * resultArray);
    static Status compute(const normal::Parameter<algorithmFPType> * parameter, engines::internal::BatchBaseImpl & engine, size_t n,
                          algorithmFPType * resultArray);
};

template <typename algorithmFPType, CpuType cpu>
using NormalKernelDefault = NormalKernel<algorithmFPType, normal::defaultDense, cpu>;

} // namespace internal
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
