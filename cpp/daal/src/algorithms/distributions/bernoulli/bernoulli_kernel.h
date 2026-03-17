/* file: bernoulli_kernel.h */
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
//  Declaration of template function that calculates bernoulli distribution.
//--

#ifndef __BERNOULLI_KERNEL_H__
#define __BERNOULLI_KERNEL_H__

#include "src/algorithms/distributions/bernoulli/bernoulli.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace bernoulli
{
namespace internal
{
/**
 *  \brief Kernel for bernoulli calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BernoulliKernel : public Kernel
{
public:
    static Status compute(algorithmFPType p, engines::BatchBase & engine, NumericTable * resultTable);
    static Status computeInt(int * resultArray, size_t n, algorithmFPType p, engines::BatchBase & engine);
    static Status computeFPType(NumericTable * resultTable, algorithmFPType p, engines::BatchBase & engine);
};

template <typename algorithmFPType, CpuType cpu>
using BernoulliKernelDefault = BernoulliKernel<algorithmFPType, bernoulli::defaultDense, cpu>;

} // namespace internal
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
