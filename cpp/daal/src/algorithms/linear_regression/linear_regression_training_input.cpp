/* file: linear_regression_training_input.cpp */
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
//  Implementation of linear regression algorithm classes.
//--
*/

#include "algorithms/linear_regression/linear_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace interface1
{
/** Default constructor */
Input::Input() : linear_model::training::Input(lastInputId + 1) {}
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for linear regression model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return linear_model::training::Input::get(linear_model::training::InputId(id));
}

/**
 * Sets an input object for linear regression model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & value)
{
    linear_model::training::Input::set(linear_model::training::InputId(id), value);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNumberOfFeatures() const
{
    return get(data)->getNumberOfColumns();
}

/**
 * Returns the number of dependent variables
 * \return Number of dependent variables
 */
size_t Input::getNumberOfDependentVariables() const
{
    return get(dependentVariables)->getNumberOfColumns();
}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Input::check(par, method));

    const Parameter * parameter = static_cast<const Parameter *>(par);

    NumericTablePtr dataTable = get(data);
    size_t nRowsInData        = dataTable->getNumberOfRows();
    size_t nColumnsInData     = dataTable->getNumberOfColumns();

    DAAL_CHECK(nRowsInData > 0, ErrorIncorrectNumberOfObservations);
    DAAL_CHECK(nColumnsInData > 0, ErrorIncorrectNumberOfFeatures);
    return s;
}

} // namespace interface1
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
