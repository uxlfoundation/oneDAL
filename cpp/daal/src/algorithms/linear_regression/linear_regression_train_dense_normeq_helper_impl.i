/* file: linear_regression_train_dense_normeq_helper_impl.i */
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__

#include "src/algorithms/linear_regression/linear_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status KernelHelper<algorithmFPType, cpu>::computeBetasImpl(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny,
                                                            algorithmFPType * b, bool inteceptFlag, bool isDualSystem) const
{
    return FinalizeKernel<algorithmFPType, cpu>::solveSystem(p, aCopy, ny, b, getInternalErrorCode(), isDualSystem);
}

template <typename algorithmFPType, CpuType cpu>
ErrorID KernelHelper<algorithmFPType, cpu>::getInternalErrorCode() const
{
    return ErrorLinearRegressionInternal;
}
} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
