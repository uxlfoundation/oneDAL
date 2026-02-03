/* file: decision_tree_regression_training_input.cpp */
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
//  Implementation of Decision tree algorithm classes.
//--
*/

#include "algorithms/decision_tree/decision_tree_regression_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

Input::Input() : algorithms::regression::training::Input(lastInputId + 1) {}
Input::Input(const Input & other) : algorithms::regression::training::Input(other) {}

NumericTablePtr Input::get(decision_tree::regression::training::InputId id) const
{
    return algorithms::regression::training::Input::get(algorithms::regression::training::InputId(id));
}

void Input::set(decision_tree::regression::training::InputId id, const data_management::NumericTablePtr & value)
{
    algorithms::regression::training::Input::set(algorithms::regression::training::InputId(id), value);
}

size_t Input::getNumberOfFeatures() const
{
    const NumericTablePtr dataTable = get(data);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

size_t Input::getNumberOfDependentVariables() const
{
    const NumericTablePtr dependentVariablesTable = get(dependentVariables);
    return dependentVariablesTable ? dependentVariablesTable->getNumberOfColumns() : 0;
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::training::Input::check(parameter, method));
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    DAAL_CHECK_EX(getNumberOfDependentVariables() == 1, ErrorIncorrectNumberOfColumns, ArgumentName, dependentVariableStr());

    services::Status s;
    const decision_tree::regression::Parameter * const par = static_cast<const decision_tree::regression::Parameter *>(parameter);
    if (par->pruning == decision_tree::reducedErrorPruning)
    {
        const NumericTablePtr dataForPruningTable = get(dataForPruning);
        DAAL_CHECK_STATUS(s, checkNumericTable(dataForPruningTable.get(), dataForPruningStr(), 0, 0, this->getNumberOfFeatures()));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(dependentVariablesForPruning).get(), dependentVariablesForPruningStr(), 0, 0,
                                               1, dataForPruningTable->getNumberOfRows()));
    }
    else
    {
        DAAL_CHECK_EX(get(dataForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, dataForPruningStr());
        DAAL_CHECK_EX(get(dependentVariablesForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName,
                      dependentVariablesForPruningStr());
    }

    return s;
}

} // namespace interface1

using interface1::Input;

} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
