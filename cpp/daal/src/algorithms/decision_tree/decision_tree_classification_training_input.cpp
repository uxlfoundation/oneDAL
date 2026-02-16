/* file: decision_tree_classification_training_input.cpp */
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

#include "algorithms/decision_tree/decision_tree_classification_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

Input::Input() : classifier::training::Input(lastInputId + 1) {}

NumericTablePtr Input::get(decision_tree::classification::training::InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Input::set(decision_tree::classification::training::InputId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Input::check(parameter, method));
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;
    decision_tree::Pruning treePruning;
    {
        auto par2 = dynamic_cast<const decision_tree::classification::interface2::Parameter *>(parameter);
        if (par2) treePruning = par2->pruning;

        if (par2 == NULL) return services::Status(ErrorNullParameterNotSupported);
    }

    if (treePruning == decision_tree::reducedErrorPruning)
    {
        const NumericTablePtr dataForPruningTable = get(dataForPruning);
        DAAL_CHECK_STATUS(s, checkNumericTable(dataForPruningTable.get(), dataForPruningStr(), 0, 0, this->getNumberOfFeatures()));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(labelsForPruning).get(), labelsForPruningStr(), 0, 0, 1, dataForPruningTable->getNumberOfRows()));
    }
    else
    {
        DAAL_CHECK_EX(get(dataForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, dataForPruningStr());
        DAAL_CHECK_EX(get(labelsForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, labelsForPruningStr());
    }

    return s;
}

} // namespace interface1

using interface1::Input;

} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
