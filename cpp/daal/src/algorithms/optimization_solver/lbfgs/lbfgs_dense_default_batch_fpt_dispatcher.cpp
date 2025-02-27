/* file: lbfgs_dense_default_batch_fpt_dispatcher.cpp */
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
//  Instantiation of LBFGS algorithm container.
//--

#include "src/algorithms/optimization_solver/lbfgs/lbfgs_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::lbfgs::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::lbfgs::defaultDense)

namespace optimization_solver
{
namespace lbfgs
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, optimization_solver::lbfgs::defaultDense>::Batch(const sum_of_functions::BatchPtr & objectiveFunction)
    : parameter(objectiveFunction)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, optimization_solver::lbfgs::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : iterative_solver::Batch(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

template <>
services::SharedPtr<BatchType> DAAL_EXPORT BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}

} // namespace interface2
} // namespace lbfgs
} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
