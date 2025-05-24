/* file: df_classification_predict_types.cpp */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace prediction
{
namespace interface1
{
/** Default constructor */
Input::Input()                                = default;
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for making decision forest model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns an input object for making decision forest model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
decision_forest::classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<decision_forest::classification::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making decision forest model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Sets an input object for making decision forest model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::ModelInputId id, const decision_forest::classification::ModelPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks an input object for making decision forest model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(classifier::prediction::data);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const decision_forest::classification::ModelPtr m = get(classifier::prediction::model);

    if (!m.get())
    {
        s.add(ErrorNullModel);
        //TODO: check input model
    }
    else
    {
        const auto nFeatures      = dataTable->getNumberOfColumns();
        const auto nFeaturesModel = m->getNFeatures();

        DAAL_CHECK(nFeaturesModel == nFeatures, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }
    return s;
}

services::Status Parameter::check() const
{
    return daal::algorithms::classifier::interface2::Parameter::check();
}

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
