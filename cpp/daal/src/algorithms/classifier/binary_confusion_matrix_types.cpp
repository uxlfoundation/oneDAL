/* file: binary_confusion_matrix_types.cpp */
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
//  Declaration of data types for computing the binary confusion matrix.
//--
*/

#include "algorithms/classifier/binary_confusion_matrix_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace binary_confusion_matrix
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_BINARY_CONFUSION_MATRIX_RESULT_ID);
Parameter::Parameter(double beta) : beta(beta) {}
Parameter::Parameter(const Parameter & other)             = default;
Parameter & Parameter::operator=(const Parameter & other) = default;

Status Parameter::check() const
{
    DAAL_CHECK_EX(beta > 0, ErrorIncorrectParameter, ParameterName, betaStr());
    return Status();
}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object of the quality metric
 * \param[in] id   Identifier of the input object
 * \return         %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object of the quality metric
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of an input object
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    NumericTablePtr predictedLabelsTable   = get(predictedLabels);
    NumericTablePtr groundTruthLabelsTable = get(groundTruthLabels);

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(predictedLabelsTable.get(), predictedLabelsStr(), unexpectedLayouts, 0, 1));

    const size_t nRows = predictedLabelsTable->getNumberOfRows();
    return checkNumericTable(groundTruthLabelsTable.get(), groundTruthLabelsStr(), unexpectedLayouts, 0, 1, nRows);
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the quality metric of the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \return          Quality metric of the classification algorithm
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the binary confusion matrix algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \param[in] value Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    NumericTablePtr confusionMatrixTable = get(confusionMatrix);
    NumericTablePtr binaryMetricsTable   = get(binaryMetrics);
    const int unexpectedLayouts          = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(confusionMatrixTable.get(), confusionMatrixStr(), unexpectedLayouts, 0, 2, 2));
    return checkNumericTable(binaryMetricsTable.get(), binaryMetricsStr(), unexpectedLayouts, 0, 6, 1);
}

} // namespace interface1
} // namespace binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal
