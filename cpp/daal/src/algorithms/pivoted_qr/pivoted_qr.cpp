/* file: pivoted_qr.cpp */
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
//  Definition of Pivoted QR common types.
//--
*/

#include "algorithms/pivoted_qr/pivoted_qr_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_PIVOTED_QR_RESULT_ID);

Parameter::Parameter(const NumericTablePtr permutedColumns) : daal::algorithms::Parameter(), permutedColumns(permutedColumns) {}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}

/**
 * Returns input object for the pivoted QR algorithm
 * \param[in] id    Identifier of the input object
 * \return          Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}
/**
 * Sets input object for the pivoted QR algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    Status s = checkNumericTable(get(data).get(), dataStr());
    if (!s)
    {
        return s;
    }
    size_t nVectors  = get(data)->getNumberOfRows();
    size_t nFeatures = get(data)->getNumberOfColumns();

    DAAL_CHECK_EX(nVectors >= nFeatures, ErrorIncorrectNumberOfRows, ArgumentName, dataStr());

    Parameter * parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    if (parameter->permutedColumns.get() != NULL)
    {
        int unexpectedLayout = (int)NumericTableIface::csrArray;

        s |= checkNumericTable(parameter->permutedColumns.get(), permutedColumnsStr(), unexpectedLayout, 0, nFeatures, 1);
    }
    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns result of the pivoted QR algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets NumericTable to store the result of the pivoted QR algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the storage NumericTable
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
* Checks the correctness of the result object
* \param[in] in     Pointer to the input objects structure
* \param[in] par    Pointer to the structure of the algorithm parameters
* \param[in] method Computation method
*/
Status Result::check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const
{
    const Input * input = static_cast<const Input *>(in);

    size_t nVectors  = input->get(data)->getNumberOfRows();
    size_t nFeatures = input->get(data)->getNumberOfColumns();

    int unexpectedLayout = (int)NumericTableIface::csrArray;

    Status s = checkNumericTable(get(matrixQ).get(), matrixQStr(), unexpectedLayout, 0, nFeatures, nVectors);
    if (!s)
    {
        return s;
    }
    s |= checkNumericTable(get(permutationMatrix).get(), permutationMatrixStr(), unexpectedLayout, 0, nFeatures, 1);
    if (!s)
    {
        return s;
    }

    s |= checkNumericTable(get(matrixR).get(), matrixRStr(), unexpectedLayout, 0, nFeatures, nFeatures);
    return s;
}

} // namespace interface1
} // namespace pivoted_qr
} // namespace algorithms
} // namespace daal
