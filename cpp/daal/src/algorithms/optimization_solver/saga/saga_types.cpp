/* file: saga_types.cpp */
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
//  Implementation of saga solver classes.
//--
*/

#include "algorithms/optimization_solver/saga/saga_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace saga
{
namespace interface2
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SAGA_RESULT_ID);

Parameter::Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations, double accuracyThreshold,
                     const data_management::NumericTablePtr batchIndices, const size_t batchSize,
                     const data_management::NumericTablePtr learningRateSequence, size_t seed)
    :

      optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold, false, batchSize),
      batchIndices(batchIndices),
      learningRateSequence(learningRateSequence),
      seed(seed),
      engine(engines::mt19937::Batch<>::create())
{}

services::Status Parameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if (!s) return s;

    if (learningRateSequence)
    {
        const size_t nRows = learningRateSequence->getNumberOfRows();
        DAAL_CHECK_EX(nRows == 1 || nRows == nIterations, services::ErrorIncorrectNumberOfRows, services::ArgumentName, "learningRateSequence");
        DAAL_CHECK_EX(learningRateSequence->getNumberOfColumns() == 1, services::ErrorIncorrectNumberOfColumns, services::ArgumentName,
                      "learningRateSequence");
    }
    if (batchIndices)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(batchIndices.get(), "batchIndices", 0, 0, 1, nIterations));
    }
    if (batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    return s;
}

Input::Input() {}
Input::Input(const Input & other) {}
Input & Input::operator=(const Input & other)
{
    return *this;
}

data_management::NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (pOpt.get()) return data_management::NumericTable::cast(pOpt->get(id));
    return data_management::NumericTablePtr();
}

void Input::set(OptionalDataId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(iterative_solver::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = super::check(par, method);
    if (!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get()) return services::Status(); //ok
    if (pOpt->size() != lastOptionalData + 1)
    {
        return services::Status(services::ErrorIncorrectOptionalInput);
    }
    data_management::NumericTablePtr pGradientsTable = data_management::NumericTable::cast(pOpt->get(gradientsTable));
    if (pGradientsTable.get())
    {
        data_management::NumericTablePtr arg = this->get(iterative_solver::inputArgument);
        DAAL_CHECK_STATUS(s,
                          data_management::checkNumericTable(pGradientsTable.get(), "gradientsTable", 0, 0, arg->getNumberOfRows(),
                                                             static_cast<const Parameter *>(par)->function->sumOfFunctionsParameter->numberOfTerms));
    }
    return s;
}

data_management::NumericTablePtr Result::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get()) return data_management::NumericTable::cast(pOpt->get(id));
    return data_management::NumericTablePtr();
}

void Result::set(OptionalDataId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(iterative_solver::optionalResult, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = super::check(input, par, method);
    if (!s || !static_cast<const Parameter *>(par)->optionalResultRequired) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (!pOpt.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }
    if (pOpt->size() != lastOptionalData + 1)
    {
        return services::Status(services::ErrorIncorrectOptionalResult);
    }

    return s;
}

} // namespace interface2
} // namespace saga
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
