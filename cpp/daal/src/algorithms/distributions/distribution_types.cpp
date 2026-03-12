/* file: distribution_types.cpp */
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

//++
//  Implementation of distribution types.
//--

#include "src/algorithms/distributions/distribution_types.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/engines/mt19937/mt19937_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace interface1
{
ParameterBase::ParameterBase() : engine(engines::mt19937::Batch<>::create()) {}
ParameterBase::~ParameterBase() {}

Input::Input() : daal::algorithms::Input(1) {}
Input::~Input() {}

Input::Input(const Input & other) : daal::algorithms::Input(other) {}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);

    const int expectedLayouts = (int)data_management::NumericTableIface::soa | (int)data_management::NumericTableIface::aos;
    return data_management::checkNumericTable(get(tableToFill).get(), tableToFillStr(), 0, expectedLayouts);
}

Result::Result() : daal::algorithms::Result(1) {}
Result::~Result() {}

data_management::NumericTablePtr Result::get(ResultId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

void Result::set(ResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 1, services::ErrorIncorrectNumberOfInputNumericTables);

    const Input * algInput = static_cast<const Input *>(input);
    DAAL_CHECK(algInput, services::ErrorNullInput);

    const int expectedLayouts = (int)data_management::NumericTableIface::soa | (int)data_management::NumericTableIface::aos;
    return data_management::checkNumericTable(get(randomNumbers).get(), randomNumbersStr(), 0, expectedLayouts);
}

} // namespace interface1
} // namespace distributions
} // namespace algorithms
} // namespace daal
