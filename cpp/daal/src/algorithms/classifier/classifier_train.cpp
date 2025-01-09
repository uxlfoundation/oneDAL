/* file: classifier_train.cpp */
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
//  Implementation of classifier training methods.
//--
*/

#include "algorithms/classifier/classifier_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/numeric_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace interface2
{
Parameter::Parameter(size_t nClasses) : nClasses(nClasses), resultsToEvaluate(computeClassLabels) {}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nClasses > 0, services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    DAAL_CHECK_EX(resultsToEvaluate != 0, services::ErrorIncorrectParameter, services::ParameterName, resultsToEvaluateStr())
    return services::Status();
}
} // namespace interface2

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_TRAINING_RESULT_ID);

InputIface::InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
InputIface::InputIface(const InputIface & other)             = default;
InputIface & InputIface::operator=(const InputIface & other) = default;

Input::Input(size_t nElements) : InputIface(nElements) {}
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

size_t Input::getNumberOfFeatures() const
{
    return get(classifier::training::data)->getNumberOfColumns();
}

/**
 * Returns the input object in the training stage of the classification algorithm
 * \param[in] id   Identifier of the input object, \ref InputId
 * \return         Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

/**
 * Sets the input object in the training stage of the classification algorithm
 * \param[in] id    Identifier of the input object, \ref InputId
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s; // Error status
    bool flag = false;  // Flag indicates error in table of labels

    data_management::NumericTablePtr dataTable = get(data);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows                           = dataTable->getNumberOfRows();
    data_management::NumericTablePtr labelsTable = get(labels);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(labelsTable.get(), labelsStr(), 0, 0, 1, nRows));

    data_management::NumericTablePtr weightsTable = get(weights);
    if (weightsTable)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(weightsTable.get(), weightsStr(), 0, 0, 1, nRows));
    }

    if (parameter != NULL)
    {
        const daal::algorithms::classifier::interface2::Parameter * algParameter2 =
            dynamic_cast<const daal::algorithms::classifier::interface2::Parameter *>(parameter);
        if (algParameter2 != NULL)
        {
            DAAL_CHECK_EX((algParameter2->nClasses > 1) && (algParameter2->nClasses < INT_MAX), services::ErrorIncorrectParameter,
                          services::ParameterName, nClassesStr());
            int nClasses = static_cast<int>(algParameter2->nClasses);

            data_management::BlockDescriptor<int> yBD;
            const_cast<data_management::NumericTable *>(labelsTable.get())->getBlockOfRows(0, nRows, data_management::readOnly, yBD);
            const int * const dy = yBD.getBlockPtr();
            for (size_t i = 0; i < nRows; ++i)
            {
                flag |= (dy[i] >= nClasses);
            }
            if (flag)
            {
                DAAL_CHECK_STATUS(s, services::Status(services::ErrorIncorrectClassLabels));
            }
            const_cast<data_management::NumericTable *>(labelsTable.get())->releaseBlockOfRows(yBD);
        }
        else
        {
            s = services::Status(services::ErrorNullParameterNotSupported);
        }
    }

    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}
Result::Result(const size_t n) : daal::algorithms::Result(n) {}

/**
 * Returns the model trained with the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \return          Model trained with the classification algorithm
 */
classifier::ModelPtr Result::get(ResultId id) const
{
    return classifier::Model::cast(Argument::get(id));
}

/**
 * Sets the result of the training stage of the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \param[in] value Pointer to the training result
 */
void Result::set(ResultId id, const classifier::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(input, parameter);
}

services::Status Result::checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const
{
    daal::algorithms::classifier::ModelPtr m = get(model);
    DAAL_CHECK(m, services::ErrorNullModel);
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace classifier
} // namespace algorithms
} // namespace daal
