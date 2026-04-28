/* file: implicit_als_train_input.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace interface1
{
Input::Input() : daal::algorithms::Input(lastModelInputId + 1) {}

/**
 * Returns the input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input initial model object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
ModelPtr Input::get(ModelInputId id) const
{
    return services::staticPointerCast<Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input initial model object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(ModelInputId id, const ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of users equal to the number of rows in the input data set
 * \return Number of users
 */
size_t Input::getNumberOfUsers() const
{
    return get(data)->getNumberOfRows();
}

/**
 * Returns the number of items equal to the number of columns in the input data set
 * \return Number of items
 */
size_t Input::getNumberOfItems() const
{
    return get(data)->getNumberOfColumns();
}

/**
 * Checks the parameters and input objects for the implicit ALS training algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(parameter, ErrorNullParameterNotSupported);
    const Parameter * alsParameter = static_cast<const Parameter *>(parameter);
    const size_t nFactors          = alsParameter->nFactors;

    services::Status s;
    if (method == defaultDense)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr()));
    }
    else
    {
        const int expectedLayout = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr(), 0, expectedLayout));
    }

    NumericTablePtr dataTable = get(data);
    const size_t nUsers       = dataTable->getNumberOfRows();
    const size_t nItems       = dataTable->getNumberOfColumns();
    ModelPtr model            = get(inputModel);
    DAAL_CHECK(model, ErrorNullModel);

    const int unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getUsersFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors, nUsers));
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getItemsFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors, nItems));
    return s;
}

} // namespace interface1
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
