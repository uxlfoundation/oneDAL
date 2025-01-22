/* file: mse_types.cpp */
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
//  Implementation of mse classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/mse_types.h"
#include "data_management/data/numeric_table.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace interface2
{
/**
 * Constructs the parameter of Mean squared error objective function
 * \param[in] numberOfTerms    The number of terms in the function
 * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                               a batch of indices used to compute the function results, e.g.,
                               value of the sum of the functions. If no indices are provided,
                               all terms will be used in the computations.
 * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
 */
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices, const DAAL_UINT64 resultsToCompute)
    : sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute),
      interceptFlag(true),
      penaltyL1(new HomogenNumericTable<float>(1, 1, NumericTableIface::doAllocate, 0)),
      penaltyL2(new HomogenNumericTable<float>(1, 1, NumericTableIface::doAllocate, 0))
{}

/**
 * Constructs a Parameter by copying input objects and parameters of another Parameter
 * \param[in] other An object to be used as the source object to initialize from
 */
Parameter::Parameter(const Parameter & other)
    : sum_of_functions::Parameter(other), interceptFlag(other.interceptFlag), penaltyL1(other.penaltyL1), penaltyL2(other.penaltyL2)
{}

/**
 * Assigns a Parameter by copying input objects and parameters of another Parameter
 * \param[in] other An object to be used as the source object to copy from
 */
Parameter & Parameter::operator=(const Parameter & other)
{
    sum_of_functions::Parameter::operator=(other);
    this->interceptFlag = other.interceptFlag;
    this->penaltyL1     = other.penaltyL1;
    this->penaltyL2     = other.penaltyL2;
    return *this;
}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    return sum_of_functions::Parameter::check();
}

/** Default constructor */
Input::Input() : sum_of_functions::Input(lastInputId + 1) {}

Input::Input(const Input & other) : sum_of_functions::Input(other) {}

Input & Input::operator=(const Input & other)
{
    sum_of_functions::Input::operator=(other);
    return *this;
}

/**
 * Sets one input object for Mean squared error objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Mean squared error objective function
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
* Returns optional input of the iterative solver algorithm
* \param[in] id    Identifier of the optional input data
* \return          %Input data that corresponds to the given identifier
*/
algorithms::OptionalArgumentPtr Input::get(OptionalInputId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets optional input for the iterative solver algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns input NumericTable containing optional data
* \param[in] id    Identifier of the input numeric table
* \return          %Input numeric table that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(optimization_solver::mse::optionalArgument);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

/**
* Sets optional input for the algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(OptionalDataId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(optimization_solver::mse::optionalArgument);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(optimization_solver::mse::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    sum_of_functions::Input::check(par, method);

    DAAL_CHECK(Argument::size() == 3, services::ErrorIncorrectNumberOfInputNumericTables);

    services::Status s = checkNumericTable(get(data).get(), dataStr(), 0, 0);
    if (!s) return s;
    const size_t nColsInData = get(data)->getNumberOfColumns();
    const size_t nRowsInData = get(data)->getNumberOfRows();

    s = checkNumericTable(get(dependentVariables).get(), dependentVariablesStr(), 0, 0, 0, nRowsInData);

    s |= checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1, nColsInData + 1);

    return s;
}

} // namespace interface2

} // namespace mse
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
