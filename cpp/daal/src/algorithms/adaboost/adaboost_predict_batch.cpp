/* file: adaboost_predict_batch.cpp */
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
//  Implementation of the interface for AdaBoost model-based prediction
//--
*/

#include "algorithms/boosting/adaboost_predict_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace prediction
{
namespace interface2
{
Input::Input()                                = default;
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;
Input::~Input()                               = default;

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the AdaBoost algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
adaboost::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<adaboost::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the AdaBoost algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const adaboost::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s = classifier::prediction::Input::check(parameter, method);
    if (!s)
    {
        return s;
    }

    adaboost::ModelPtr m = get(classifier::prediction::model);
    DAAL_CHECK(m->getNumberOfWeakLearners() > 0, ErrorModelNotFullInitialized);

    s |= checkNumericTable(m->getAlpha().get(), alphaStr());
    if (!s)
    {
        return Status(ErrorModelNotFullInitialized);
    }

    DAAL_CHECK(m->getNumberOfWeakLearners() == m->getAlpha()->getNumberOfRows(), ErrorIncorrectSizeOfModel);
    return s;
}
} // namespace interface2
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal
