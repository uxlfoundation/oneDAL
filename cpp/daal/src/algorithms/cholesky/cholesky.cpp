/* file: cholesky.cpp */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "algorithms/cholesky/cholesky_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CHOLESKY_RESULT_ID);

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
 * Returns input NumericTable of the Cholesky algorithm
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input numeric table that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the Cholesky algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the Cholesky algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    NumericTablePtr inTable = get(data);

    DAAL_CHECK(inTable.get(), ErrorNullInputNumericTable);
    DAAL_CHECK(inTable->getNumberOfRows(), ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(inTable->getNumberOfColumns(), ErrorIncorrectNumberOfFeatures);
    DAAL_CHECK(inTable->getNumberOfColumns() == inTable->getNumberOfRows(), ErrorIncorrectSizeOfInputNumericTable);

    return Status();
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns result of the Cholesky algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the Cholesky algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the Cholesky algorithm
 * \param[in] input   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    NumericTablePtr resTable = get(choleskyFactor);

    DAAL_CHECK(resTable.get(), ErrorNullInputNumericTable);
    DAAL_CHECK(resTable->getNumberOfRows() != 0, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(resTable->getNumberOfColumns() != 0, ErrorIncorrectNumberOfFeatures);

    NumericTableIface::StorageLayout rLayout = resTable->getDataLayout();

    Input * algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

    DAAL_CHECK((resTable->getNumberOfColumns() == algInput->get(data)->getNumberOfColumns())
                   && (resTable->getNumberOfColumns() == resTable->getNumberOfRows()),
               ErrorIncorrectSizeOfOutputNumericTable);

    const int rLayoutInt = (int)rLayout;
    DAAL_CHECK(rLayout != NumericTableIface::csrArray, ErrorIncorrectTypeOfOutputNumericTable);
    return Status();
}

} // namespace interface1
} // namespace cholesky
} // namespace algorithms
} // namespace daal
