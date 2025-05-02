/* file: logistic_loss_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of logloss calculation algorithm container.
//--

#include "src/algorithms/objective_function/logistic_loss/logistic_loss_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::logistic_loss::interface2::BatchContainer, batch, DAAL_FPTYPE,
                                      optimization_solver::logistic_loss::defaultDense)
namespace optimization_solver
{
namespace logistic_loss
{
namespace interface2
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::logistic_loss::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(size_t numberOfTerms) : sum_of_functions::Batch(numberOfTerms, &input, new ParameterType(numberOfTerms))
{
    initialize();
    _par = sumOfFunctionsParameter;
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other)
    : sum_of_functions::Batch(other.parameter().numberOfTerms, &input, new ParameterType(other.parameter())), input(other.input)
{
    initialize();
    _par = sumOfFunctionsParameter;
}

template <>
services::SharedPtr<BatchType> DAAL_EXPORT BatchType::create(size_t numberOfTerms)
{
    return services::SharedPtr<BatchType>(new BatchType(numberOfTerms));
}

} // namespace interface2
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
