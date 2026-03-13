/* file: lbfgs_dense_default_kernel.h */
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
//  Declaration of template function that computes LBFGS.
//--

#ifndef __LBFGS_DENSE_DEFAULT_KERNEL_H__
#define __LBFGS_DENSE_DEFAULT_KERNEL_H__

#include "src/algorithms/optimization_solver/lbfgs/lbfgs_base.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{
/**
 *  \brief Kernel for LBFGS computation
 */
template <typename algorithmFPType, CpuType cpu>
class LBFGSKernel<algorithmFPType, defaultDense, cpu> : public Kernel
{
public:
    services::Status compute(NumericTable * correctionPairsInput, NumericTable * correctionIndicesInput, NumericTable * inputArgument,
                             NumericTable * averageArgLIterInput, OptionalArgument * optionalArgumentInput, NumericTable * correctionPairsResult,
                             NumericTable * correctionIndicesResult, NumericTable * minimum, NumericTable * nIterationsNT,
                             NumericTable * averageArgLIterResult, OptionalArgument * optionalArgumentResult, Parameter * parameter,
                             engines::BatchBase & engine);
};

} // namespace internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
