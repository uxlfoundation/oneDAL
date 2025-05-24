/* file: iterative_solver_types.cpp */
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
//  Implementation of iterative solver classes.
//--
*/

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "data_management/data/numeric_table.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
namespace interface2
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ITERATIVE_SOLVER_RESULT_ID);

Parameter::Parameter(const sum_of_functions::BatchPtr & function_, const size_t nIterations_, const double accuracyThreshold_, bool optionalResultReq,
                     size_t batchSize_)
    : nIterations(nIterations_), accuracyThreshold(accuracyThreshold_), optionalResultRequired(optionalResultReq), batchSize(batchSize_)
{
    if (function_)
    {
        function = function_->clone();
    }
}

Parameter::Parameter(const Parameter & other)
    : nIterations(other.nIterations),
      accuracyThreshold(other.accuracyThreshold),
      optionalResultRequired(other.optionalResultRequired),
      batchSize(other.batchSize)
{
    if (other.function)
    {
        function = other.function->clone();
    }
}

Parameter & Parameter::operator=(const Parameter & other)
{
    nIterations            = other.nIterations;
    accuracyThreshold      = other.accuracyThreshold;
    optionalResultRequired = other.optionalResultRequired;
    batchSize              = other.batchSize;

    if (other.function)
    {
        function = other.function->clone();
    }
    else
    {
        function.reset();
    }

    return *this;
}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(function.get(), ErrorIncorrectParameter, ArgumentName, "function");
    DAAL_CHECK_EX(function->sumOfFunctionsParameter, ErrorNullParameterNotSupported, ArgumentName, "sumOfFunctionsParameter");
    DAAL_CHECK_EX(function->sumOfFunctionsInput, ErrorNullInput, ArgumentName, "sumOfFunctionsInput");
    DAAL_CHECK_EX(accuracyThreshold >= 0, ErrorIncorrectParameter, ArgumentName, "accuracyThreshold");
    return services::Status();
}

Input::Input() : daal::algorithms::Input(lastOptionalInputId + 1) {}
Input::Input(const Input & other) : daal::algorithms::Input(other) {}
Input & Input::operator=(const Input & other)
{
    daal::algorithms::Input::operator=(other);
    return *this;
}

data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(InputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

algorithms::OptionalArgumentPtr Input::get(OptionalInputId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(OptionalInputId id, const algorithms::OptionalArgumentPtr & ptr)
{
    Argument::set(id, ptr);
}

NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Input::set(OptionalDataId id, const NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get())
    {
        // pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    if (this->size() != 2) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);
    return checkNumericTable(get(inputArgument).get(), inputArgumentStr(), 0, 0, 1);
}

data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(ResultId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

algorithms::OptionalArgumentPtr Result::get(OptionalResultId id) const
{
    return services::staticPointerCast<algorithms::OptionalArgument, data_management::SerializationIface>(Argument::get(id));
}

void Result::set(OptionalResultId id, const algorithms::OptionalArgumentPtr & ptr)
{
    Argument::set(id, ptr);
}

NumericTablePtr Result::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Result::set(OptionalDataId id, const NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (!pOpt.get())
    {
        // pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalResult, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    if (Argument::size() != 3) return services::Status(services::ErrorIncorrectNumberOfOutputNumericTables);

    const Input * algInput = static_cast<const Input *>(input);
    size_t nRows           = algInput->get(inputArgument)->getNumberOfRows();

    services::Status s = checkNumericTable(get(minimum).get(), minimumStr(), 0, 0, 1, nRows);
    if (!s) return s;
    return checkNumericTable(get(nIterations).get(), nIterationsStr(), 0, 0, 1, 1);
}

} // namespace interface2
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
