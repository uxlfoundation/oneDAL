/* file: covariance_partialresult.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "algorithms/covariance/covariance_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID);
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Gets the number of columns in the partial result of the correlation or variance-covariance matrix algorithm
 * \return Number of columns in the partial result
 */
size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(crossProduct));
    if (ntPtr)
    {
        return ntPtr->getNumberOfColumns();
    }
    return 0;
}

/**
 * Returns the partial result of the correlation or variance-covariance matrix algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result of the correlation or variance-covariance matrix algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Check correctness of the partial result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const InputIface * algInput = static_cast<const InputIface *>(input);
    size_t nFeatures            = algInput->getNumberOfFeatures();
    return checkImpl(nFeatures);
}

/**
 * Check the correctness of PartialResult object
 * \param[in] parameter Pointer to the structure of the parameters of the algorithm
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * parameter, int method) const
{
    size_t nFeatures = getNumberOfFeatures();
    return checkImpl(nFeatures);
}

services::Status PartialResult::checkImpl(size_t nFeatures) const
{
    services::Status s;

    constexpr int unexpectedLayout = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayout, 0, 1, 1);
    if (!s) return s;

    s |= checkNumericTable(get(crossProduct).get(), crossProductCorrelationStr(), unexpectedLayout, 0, nFeatures, nFeatures);
    if (!s) return s;

    s |= checkNumericTable(get(sum).get(), sumStr(), unexpectedLayout, 0, nFeatures, 1);
    return s;
}

} //namespace interface1

} //namespace covariance
} // namespace algorithms
} // namespace daal
