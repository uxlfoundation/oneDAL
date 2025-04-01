/* file: quantiles.cpp */
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
//  Implementation of quantiles algorithm and types methods.
//--
*/

#include "algorithms/quantiles/quantiles_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_QUANTILES_RESULT_ID);
Parameter::Parameter(const NumericTablePtr quantileOrders) : daal::algorithms::Parameter(), quantileOrders(quantileOrders)
{
    Status s;
    if (quantileOrders.get() == NULL)
    {
        this->quantileOrders = HomogenNumericTable<double>::create(1, 1, NumericTableIface::doAllocate, 0.5, &s);
        if (!s) return;
    }
}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for the quantiles algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the quantiles algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check the correctness of the %Input object
 * \param[in] parameter Pointer to the parameters structure
 * \param[in] method    Algorithm computation method
 */
Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);

    Status s = checkNumericTable(algParameter->quantileOrders.get(), quantileOrdersStr(), 0, 0, 0, 1);

    s |= checkNumericTable(get(data).get(), dataStr());
    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the final result of the quantiles algorithm
 * \param[in] id   Identifier of the final result, \ref ResultId
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the Result object of the quantiles algorithm
 * \param[in] id        Identifier of the Result object
 * \param[in] value     Pointer to the Result object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] in     Pointer to the object
 * \param[in] par    Pointer to the parameters structure
 * \param[in] method Algorithm computation method
 */
Status Result::check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const
{
    const Input * input         = static_cast<const Input *>(in);
    const Parameter * parameter = static_cast<const Parameter *>(par);

    Status s = checkNumericTable(parameter->quantileOrders.get(), quantileOrdersStr(), 0, 0, 0, 1);
    if (!s) return s;

    size_t nVectors  = input->get(data)->getNumberOfColumns();
    size_t nFeatures = parameter->quantileOrders->getNumberOfColumns();

    int unexpectedLayouts = (int)NumericTableIface::csrArray | (int)NumericTableIface::upperPackedTriangularMatrix
                            | (int)NumericTableIface::lowerPackedTriangularMatrix | (int)NumericTableIface::upperPackedSymmetricMatrix
                            | (int)NumericTableIface::lowerPackedSymmetricMatrix;

    s |= checkNumericTable(get(quantiles).get(), quantilesStr(), unexpectedLayouts, 0, nFeatures, nVectors);
    return s;
}

} // namespace interface1
} // namespace quantiles
} // namespace algorithms
} // namespace daal
