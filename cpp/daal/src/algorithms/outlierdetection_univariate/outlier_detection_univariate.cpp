/* file: outlier_detection_univariate.cpp */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#include "algorithms/outlier_detection/outlier_detection_univariate_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_OUTLIER_DETECTION_UNIVARIATE_RESULT_ID);

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for the univariate outlier detection algorithm
 * \param[in] id    Identifier of the %input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the univariate outlier detection algorithm
 * \param[in] id    Identifier of the %input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks input objects for the univariate outlier detection algorithm
 * \param[in] par     Parameters of the algorithm
 * \param[in] method  univariate outlier detection computation method
      */
services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    return checkNumericTable(get(data).get(), dataStr());
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns a result of the univariate outlier detection algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a result of the univariate outlier detection algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result object of the univariate outlier detection algorithm
 * \param[in] input   Pointer to the  %input objects for the algorithm
 * \param[in] par     Pointer to the parameters of the algorithm
 * \param[in] method  univariate outlier detection computation method
 * \return             Status of checking
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Input * algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    size_t nVectors  = algInput->get(data)->getNumberOfRows();

    int unexpectedLayouts = packed_mask;
    return checkNumericTable(get(weights).get(), weightsStr(), unexpectedLayouts, 0, nFeatures, nVectors);
}

} // namespace interface1
} // namespace univariate_outlier_detection
} // namespace algorithms
} // namespace daal
