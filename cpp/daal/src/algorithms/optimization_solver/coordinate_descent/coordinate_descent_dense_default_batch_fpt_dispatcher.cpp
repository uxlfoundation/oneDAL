/* file: coordinate_descent_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of coordinate_descent calculation algorithm container.
//--

#include "src/algorithms/optimization_solver/coordinate_descent/coordinate_descent_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::coordinate_descent::BatchContainer, batch, DAAL_FPTYPE,
                                      optimization_solver::coordinate_descent::defaultDense)

namespace optimization_solver
{
namespace coordinate_descent
{
namespace interface1
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::coordinate_descent::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const sum_of_functions::BatchPtr & objectiveFunction)
{
    _par = new algorithms::optimization_solver::coordinate_descent::Parameter(objectiveFunction);
    initialize();
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : iterative_solver::Batch(other), input(other.input)
{
    _par = new algorithms::optimization_solver::coordinate_descent::Parameter(other.parameter());
    initialize();
}

template <>
services::SharedPtr<BatchType> DAAL_EXPORT BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
