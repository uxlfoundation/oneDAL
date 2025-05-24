/* file: coordinate_descent_types.cpp */
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
//  Implementation of coordinate_descent solver classes.
//--
*/

#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace coordinate_descent
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_COORDINATE_DESCENT_RESULT_ID);

Parameter::Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations, double accuracyThreshold, size_t seed)
    :

      optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold, false, 1),
      seed(seed),
      engine(engines::mt19937::Batch<>::create()),
      selection(cyclic),
      positive(false),
      skipTheFirstComponents(false)
{}

services::Status Parameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if (!s) return s;

    if (batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    return s;
}

Input::Input() {}
Input::Input(const Input & other) {}
Input & Input::operator=(const Input & other)
{
    return *this;
};

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s;
    if (this->size() != 2) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    s = data_management::checkNumericTable(get(iterative_solver::inputArgument).get(), inputArgumentStr(), 0, 0);

    if (!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get()) return services::Status(); //ok

    return s;
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    services::Status s; // = super::check(input, par, method);
    if (!s || !static_cast<const Parameter *>(par)->optionalResultRequired) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (!pOpt.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }

    return s;
}

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
