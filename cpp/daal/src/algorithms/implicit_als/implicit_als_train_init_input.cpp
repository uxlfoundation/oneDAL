/* file: implicit_als_train_init_input.cpp */
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

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
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
namespace init
{
namespace interface1
{
Input::Input(size_t nElements) : daal::algorithms::Input(nElements) {}

/**
 * Returns an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of items, that is, the number of columns in the input data set
 * \return Number of items
 */
size_t Input::getNumberOfItems() const
{
    return get(data)->getNumberOfRows();
}

/**
 * Checks the input objects and parameters of the implicit ALS initialization algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    if (method == defaultDense)
    {
        return checkNumericTable(get(data).get(), dataStr());
    }
    const int expectedLayout = (int)NumericTableIface::csrArray;
    return checkNumericTable(get(data).get(), dataStr(), 0, expectedLayout);
}

} // namespace interface1
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
