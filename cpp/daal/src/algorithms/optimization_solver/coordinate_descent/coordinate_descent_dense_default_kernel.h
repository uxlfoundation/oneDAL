/* file: coordinate_descent_dense_default_kernel.h */
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
//  Declaration of template function that calculate coordinate_descent.
//--

#ifndef __COORDINATE_DESCENT_DENSE_DEFAULT_KERNEL_H__
#define __COORDINATE_DESCENT_DENSE_DEFAULT_KERNEL_H__

#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_math.h"
#include "src/data_management/service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace coordinate_descent
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class CoordinateDescentKernel : public Kernel
{
public:
    services::Status compute(NumericTable * inputArgument, NumericTable * minimum, NumericTable * nIterations, Parameter * parameter,
                             engines::BatchBase & engine, optimization_solver::objective_function::ResultPtr & hesGr,
                             optimization_solver::objective_function::ResultPtr & prox);
};

} // namespace internal

} // namespace coordinate_descent

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
