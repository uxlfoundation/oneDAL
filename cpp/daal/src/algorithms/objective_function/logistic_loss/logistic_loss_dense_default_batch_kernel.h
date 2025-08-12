/* file: logistic_loss_dense_default_batch_kernel.h */
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
//  Declaration of template function that calculate logloss.
//--

#ifndef __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"
#include "src/algorithms/kernel.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
class LogLossKernel : public Kernel
{
public:
    services::Status compute(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value,
                             NumericTable * hessian, NumericTable * gradient, NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                             NumericTable * lipschitzConstant, Parameter * parameter);
    static void applyBeta(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb, size_t nRows, size_t nCols, bool bIntercept);

    static void sigmoid(const algorithmFPType * f, algorithmFPType * s, size_t n);
    static void sigmoid_clipped(const algorithmFPType * f, algorithmFPType * s, size_t n);

protected:
    services::Status doCompute(const NumericTable * dataNT, const NumericTable * dependentVariablesNT, size_t n, size_t p, NumericTable * betaNT,
                               NumericTable * valueNT, NumericTable * hessianNT, NumericTable * gradientNT, NumericTable * nonSmoothTermValue,
                               NumericTable * proximalProjection, NumericTable * lipschitzConstant, Parameter * parameter);

private:
    TArrayScalable<algorithmFPType, cpu> _aX;
    TArrayScalable<algorithmFPType, cpu> _aY;
};

} // namespace internal

} // namespace logistic_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
