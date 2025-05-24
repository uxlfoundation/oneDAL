/* file: adagrad_types.cpp */
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
//  Implementation of adagrad solver classes.
//--
*/

#include "algorithms/optimization_solver/adagrad/adagrad_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace interface2
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ADAGRAD_RESULT_ID);

Parameter::Parameter(const sum_of_functions::BatchPtr & function_, size_t nIterations_, double accuracyThreshold_,
                     data_management::NumericTablePtr batchIndices_, const size_t batchSize_, data_management::NumericTablePtr learningRate_,
                     double degenerateCasesThreshold_, size_t seed_)
    :

      optimization_solver::iterative_solver::Parameter(function_, nIterations_, accuracyThreshold_, false, batchSize_),
      batchIndices(batchIndices_),
      learningRate(learningRate_),
      degenerateCasesThreshold(degenerateCasesThreshold_),
      seed(seed_),
      engine(engines::mt19937::Batch<>::create())
{}

services::Status Parameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if (!s) return s;

    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(learningRate.get(), learningRateStr(), 0, 0, 1, 1));

    if (batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    if (batchIndices) return data_management::checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
    return s;
}

Input::Input() {}
Input::Input(const Input & other) {}
Input & Input::operator=(const Input & other)
{
    return *this;
};

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

static services::Status checkGradientSquareSumData(const daal::algorithms::Input * input, const data_management::SerializationIfacePtr & pItem,
                                                   bool bInput)
{
    const services::ErrorDetailID det      = bInput ? services::OptionalInput : services::OptionalResult;
    data_management::NumericTablePtr pData = data_management::NumericTable::cast(pItem);
    DAAL_CHECK_EX(pData.get(), bInput ? services::ErrorIncorrectOptionalInput : services::ErrorIncorrectOptionalResult, det, gradientSquareSumStr());
    const Input * algInput = static_cast<const Input *>(input);
    auto arg               = algInput->get(iterative_solver::inputArgument);
    DAAL_CHECK_EX(pData->getNumberOfColumns() == arg->getNumberOfColumns(), services::ErrorIncorrectNumberOfColumns, det, gradientSquareSumStr());
    DAAL_CHECK_EX(pData->getNumberOfRows() == arg->getNumberOfRows(), services::ErrorIncorrectNumberOfRows, det, gradientSquareSumStr())
    return services::Status();
}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = super::check(par, method);
    if (!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get()) return services::Status(); //ok
    DAAL_CHECK(pOpt->size() == lastOptionalData + 1, services::ErrorIncorrectOptionalInput);

    auto pItem = pOpt->get(gradientSquareSum);
    if (pItem.get())
    {
        DAAL_CHECK_STATUS(s, checkGradientSquareSumData(this, pItem, true));
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
    DAAL_CHECK(pOpt.get(), services::ErrorNullOptionalResult);
    DAAL_CHECK(pOpt->size() == lastOptionalData + 1, services::ErrorIncorrectOptionalResult);
    auto pItem = pOpt->get(gradientSquareSum);
    DAAL_CHECK(pItem.get(), services::ErrorNullOptionalResult);
    s |= checkGradientSquareSumData(input, pItem, false);
    return s;
}

} // namespace interface2
} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
