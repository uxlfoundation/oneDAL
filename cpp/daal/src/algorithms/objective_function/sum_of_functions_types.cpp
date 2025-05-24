/* file: sum_of_functions_types.cpp */
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

#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"
#include "data_management/data/numeric_table.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{
namespace interface2
{
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices, const DAAL_UINT64 resultsToCompute)
    : numberOfTerms(numberOfTerms), objective_function::Parameter(resultsToCompute), batchIndices(batchIndices), featureId(0)
{}

Parameter::Parameter(const Parameter & other)
    : numberOfTerms(other.numberOfTerms),
      objective_function::Parameter(other.resultsToCompute),
      batchIndices(other.batchIndices),
      featureId(other.featureId)
{}

Parameter & Parameter::operator=(const Parameter & other)
{
    objective_function::Parameter::operator=(other.resultsToCompute);
    this->numberOfTerms = other.numberOfTerms;
    this->batchIndices  = other.batchIndices;
    this->featureId     = other.featureId;
    return *this;
}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    if (batchIndices.get() != NULL)
        DAAL_CHECK_EX(batchIndices->getNumberOfRows() == 1, ErrorIncorrectNumberOfObservations, ArgumentName, batchIndicesStr());
    DAAL_CHECK(numberOfTerms != 0, ErrorZeroNumberOfTerms);
    return services::Status();
}

/** Default constructor */
Input::Input(size_t n) : objective_function::Input(n) {}

Input::Input(const Input & other) : objective_function::Input(other) {}

Input & Input::operator=(const Input & other)
{
    objective_function::Input::operator=(other);
    return *this;
}

/**
 * Sets one input object for Sum of functions
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Sum of functions
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    const Parameter * algParameter = static_cast<const Parameter *>(par);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

    services::Status s = checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1);

    return s;
}

} // namespace interface2
} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
