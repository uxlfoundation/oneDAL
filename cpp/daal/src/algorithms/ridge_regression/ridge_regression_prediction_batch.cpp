/* file: ridge_regression_prediction_batch.cpp */
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
//  Implementation of ridge regression algorithm classes.
//--
*/

#include "algorithms/ridge_regression/ridge_regression_predict_types.h"
#include "src/services/serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_RIDGE_REGRESSION_PREDICTION_RESULT_ID);

/** Default constructor */
Input::Input() : linear_model::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input & other) : linear_model::prediction::Input(other) {}
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(NumericTableInputId id) const
{
    return linear_model::prediction::Input::get(linear_model::prediction::NumericTableInputId(id));
}

/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
ridge_regression::ModelPtr Input::get(ModelInputId id) const
{
    return ridge_regression::Model::cast(linear_model::prediction::Input::get(linear_model::prediction::ModelInputId(id)));
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(NumericTableInputId id, const NumericTablePtr & value)
{
    linear_model::prediction::Input::set(linear_model::prediction::NumericTableInputId(id), value);
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const ridge_regression::ModelPtr & value)
{
    linear_model::prediction::Input::set(linear_model::prediction::ModelInputId(id), value);
}

Result::Result() : linear_model::prediction::Result(lastResultId + 1) {}

/**
 * Returns the result of ridge regression model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return linear_model::prediction::Result::get(linear_model::prediction::ResultId(id));
}

/**
 * Sets the result of ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    linear_model::prediction::Result::set(linear_model::prediction::ResultId(id), value);
}

} // namespace interface1
} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
