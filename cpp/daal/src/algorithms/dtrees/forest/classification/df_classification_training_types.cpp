/* file: df_classification_training_types.cpp */
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

#include "src/algorithms/dtrees/forest/classification/df_classification_training_types_result.h"
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
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_CLASSIFICATION_TRAINING_RESULT_ID);

Result::Result() : classifier::training::Result(lastResultId + 1)
{
    _impl = new Result::ResultImpl();
}

Result::~Result()
{
    delete _impl;
    _impl = nullptr;
}

Result::Result(const Result & other) : classifier::training::Result(other)
{
    _impl = new Result::ResultImpl(*other._impl);
}

daal::algorithms::decision_forest::classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::decision_forest::classification::Model, SerializationIface>(Argument::get(id));
}

void Result::set(classifier::training::ResultId id, const ModelPtr & value)
{
    Argument::set(id, value);
}

NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == lastResultId + 1, ErrorIncorrectNumberOfOutputNumericTables);
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);

    ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m.get(), ErrorNullModel);

    services::Status s;
    const daal::algorithms::decision_forest::training::interface2::Parameter * algParameter2 =
        dynamic_cast<const daal::algorithms::decision_forest::training::interface2::Parameter *>(par);
    if (algParameter2 != NULL)
    {
        if (algParameter2->resultsToCompute & decision_forest::training::computeOutOfBagError)
        {
            DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagError).get(), outOfBagErrorStr(), 0, 0, 1, 1));
        }
        if (algParameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorAccuracy)
        {
            DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagErrorAccuracy).get(), outOfBagErrorAccuracyStr(), 0, 0, 1, 1));
        }
        if (algParameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
        {
            const auto nObs = algInput->get(classifier::training::data)->getNumberOfRows();
            DAAL_CHECK_STATUS(
                s, data_management::checkNumericTable(get(outOfBagErrorPerObservation).get(), outOfBagErrorPerObservationStr(), 0, 0, 1, nObs));
        }
        if (algParameter2->resultsToCompute & decision_forest::training::computeOutOfBagErrorDecisionFunction)
        {
            const decision_forest::classification::training::Parameter * algParameter3 =
                dynamic_cast<const decision_forest::classification::training::Parameter *>(par);
            if (algParameter3 != NULL)
            {
                const size_t nClasses = algParameter3->nClasses;
                const auto nObs       = algInput->get(classifier::training::data)->getNumberOfRows();
                DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagErrorDecisionFunction).get(), outOfBagErrorDecisionFunctionStr(),
                                                                        0, 0, nClasses, nObs));
            }
            else
            {
                s = s ? s : services::Status(services::ErrorNullParameterNotSupported);
            }
        }
        if (algParameter2->varImportance != decision_forest::training::none)
        {
            const auto nFeatures = algInput->get(classifier::training::data)->getNumberOfColumns();
            DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(variableImportance).get(), variableImportanceStr(), 0, 0, nFeatures, 1));
        }
    }
    else
    {
        s = s ? s : services::Status(services::ErrorNullParameterNotSupported);
    }
    return s;
}

engines::EngineIfacePtr Result::get(ResultEngineId id) const
{
    return _impl->getEngine();
}
} // namespace interface1

namespace interface3
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    DAAL_CHECK_STATUS(s, decision_forest::training::checkImpl(*this));
    return s;
}
} // namespace interface3

} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
