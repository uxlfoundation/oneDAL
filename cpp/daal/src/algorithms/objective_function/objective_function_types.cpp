/* file: objective_function_types.cpp */
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
//  Implementation of objective function classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/objective_function_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID);
Parameter::Parameter(const DAAL_UINT64 resultsToCompute) : resultsToCompute(resultsToCompute) {}

Parameter::Parameter(const Parameter & other) : resultsToCompute(other.resultsToCompute) {}

Parameter & Parameter::operator=(const Parameter & other)
{
    if (this == &other) return *this;
    this->resultsToCompute = other.resultsToCompute;
    return *this;
}

/** Default constructor */
Input::Input(size_t n) : daal::algorithms::Input(n) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other)
{
    daal::algorithms::Input::operator=(other);
    return *this;
}

/**
 * Sets one input object for Objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Objective function
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(par != 0, services::ErrorNullParameterNotSupported);
    return checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1);
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Sets the result of the Objective function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the numeric table with the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the collection of the results of the Objective function
 * \param[in] id   Identifier of the result
 * \return         %Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Checks the result of the Objective function
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    using namespace services;

    DAAL_CHECK(Argument::size() == 9, ErrorIncorrectNumberOfArguments);

    const Input * algInput         = static_cast<const Input *>(input);
    const Parameter * algParameter = static_cast<const Parameter *>(par);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);
    const size_t nRows = algInput->get(argument)->getNumberOfRows();

    services::Status s;
    if (algParameter->resultsToCompute & value)
    {
        s = checkNumericTable(get(valueIdx).get(), valueIdxStr(), 0, 0, 1, 1);
    }
    if (algParameter->resultsToCompute & gradient)
    {
        s |= checkNumericTable(get(gradientIdx).get(), gradientIdxStr(), 0, 0, 1, nRows);
    }
    if (algParameter->resultsToCompute & hessian)
    {
        s |= checkNumericTable(get(hessianIdx).get(), hessianIdxStr(), 0, 0, nRows, nRows);
    }
    return s;
}

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
