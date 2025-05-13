/* file: kernel_function.cpp */
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
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "algorithms/kernel_function/kernel_function_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_KERNEL_FUNCTION_RESULT_ID);

ParameterBase::ParameterBase(size_t rowIndexX, size_t rowIndexY, size_t rowIndexResult, ComputationMode computationMode)
    : rowIndexX(rowIndexX), rowIndexY(rowIndexY), rowIndexResult(rowIndexResult), computationMode(computationMode)
{}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other) = default;
/**
* Returns the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the input object
*/
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

Status Input::checkCSR() const
{
    Status s;
    const int csrLayout = (int)NumericTableIface::csrArray;

    DAAL_CHECK_STATUS(s, checkNumericTable(get(X).get(), XStr(), 0, csrLayout));

    const size_t nFeaturesX = get(X)->getNumberOfColumns();

    return checkNumericTable(get(Y).get(), YStr(), 0, csrLayout, nFeaturesX);
}

Status Input::checkDense() const
{
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(X).get(), XStr()));

    const size_t nFeaturesX = get(X)->getNumberOfColumns();

    return checkNumericTable(get(Y).get(), YStr(), 0, 0, nFeaturesX);
}
/**
 * Returns the result of the kernel function algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the kernel function algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the kernel function algorithm
* \param[in] input   %Input objects of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status st;
    Input * algInput             = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    ParameterBase * algParameter = static_cast<ParameterBase *>(const_cast<daal::algorithms::Parameter *>(par));

    const size_t nRowsX         = algInput->get(X)->getNumberOfRows();
    const size_t nRowsY         = algInput->get(Y)->getNumberOfRows();
    const size_t nVectorsValues = get(values)->getNumberOfRows();

    if (nVectorsValues != nRowsX && nVectorsValues != nRowsY)
    {
        st.add(Error::create(ErrorIncorrectNumberOfRows, ParameterName, valuesStr()));
    }
    if (algParameter->rowIndexResult >= nVectorsValues)
    {
        st.add(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexResultStr()));
    }
    if (algParameter->rowIndexX >= nRowsX)
    {
        st.add(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexXStr()));
    }
    if (algParameter->rowIndexY >= nRowsY)
    {
        st.add(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexYStr()));
    }
    return st;
}

} // namespace interface1
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
