/* file: regression_training_input.cpp */
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
//  Implementation of the class defining the input objects
//  of the regression training algorithm
//--
*/

#include "algorithms/regression/regression_training_types.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;
Input::Input(size_t nElements) : daal::algorithms::Input(nElements) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other) = default;

data_management::NumericTablePtr Input::get(InputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    const NumericTablePtr dataTable              = get(data);
    const NumericTablePtr dependentVariableTable = get(dependentVariables);

    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    size_t nRowsInData = dataTable->getNumberOfRows();

    return checkNumericTable(dependentVariableTable.get(), dependentVariableStr(), 0, 0, 0, nRowsInData);
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace algorithms
} // namespace daal
