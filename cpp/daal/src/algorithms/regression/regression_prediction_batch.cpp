/* file: regression_prediction_batch.cpp */
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
//  Implementation of the regression algorithm classes.
//--
*/

#include "algorithms/regression/regression_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_REGRESSION_PREDICTION_RESULT_ID);

Input::Input(size_t nElements) : daal::algorithms::Input(nElements) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other) = default;

NumericTablePtr Input::get(NumericTableInputId id) const
{
    return NumericTable::cast(Argument::get(id));
}

regression::ModelPtr Input::get(ModelInputId id) const
{
    return staticPointerCast<regression::Model, SerializationIface>(Argument::get(id));
}

void Input::set(NumericTableInputId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

void Input::set(ModelInputId id, const regression::ModelPtr & value)
{
    Argument::set(id, value);
}

Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    const NumericTablePtr dataTable = get(data);
    Status s;
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    const regression::ModelConstPtr m = get(model);
    DAAL_CHECK(m, ErrorNullModel);

    DAAL_CHECK_EX(m->getNumberOfFeatures() == dataTable->getNumberOfColumns(), ErrorIncorrectNumberOfFeatures, services::ArgumentName, dataStr());
    return s;
}

Result::Result(size_t nElements) : daal::algorithms::Result(nElements) {}

NumericTablePtr Result::get(ResultId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    NumericTablePtr predictionTable = get(prediction);
    const Input * in                = static_cast<const Input *>(input);

    size_t nRowsInData = in->get(data)->getNumberOfRows();

    return checkNumericTable(predictionTable.get(), predictionStr(), 0, 0, 0, nRowsInData);
}
} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace algorithms
} // namespace daal
