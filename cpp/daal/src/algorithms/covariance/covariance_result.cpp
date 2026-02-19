/* file: covariance_result.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "algorithms/covariance/covariance_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_COVARIANCE_RESULT_ID);
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
    * Returns the final result of the correlation or variance-covariance matrix algorithm
    * \param[in] id   Identifier of the result, \ref ResultId
    * \return Final result that corresponds to the given identifier
    */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
    * Sets the result of the correlation or variance-covariance matrix algorithm
    * \param[in] id        Identifier of the result
    * \param[in] value     Pointer to the object
    */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
    * Check correctness of the result
    * \param[in] partialResult     Pointer to the partial result arguments structure
    * \param[in] parameter         Pointer to the structure of the parameters of the algorithm
    * \param[in] method            Computation method
    */
services::Status Result::check(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter, int method) const
{
    const PartialResult * pres     = static_cast<const PartialResult *>(partialResult);
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFeatures               = pres->getNumberOfFeatures();

    return checkImpl(nFeatures, algParameter->outputMatrixType);
}

/**
    * Check correctness of the result
    * \param[in] input     Pointer to the structure with input objects
    * \param[in] parameter Pointer to the structure of algorithm parameters
    * \param[in] method    Computation method
    */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);

    size_t nFeatures = (static_cast<const InputIface *>(input))->getNumberOfFeatures();

    return checkImpl(nFeatures, algParameter->outputMatrixType);
}

services::Status Result::checkImpl(size_t nFeatures, OutputMatrixType outputMatrixType) const
{
    constexpr int unexpectedLayout = (int)NumericTableIface::csrArray;
    services::Status s;
    if (outputMatrixType == covarianceMatrix)
    {
        /* Check covariance matrix */
        s |= checkNumericTable(get(covariance).get(), covarianceStr(), unexpectedLayout, 0, nFeatures, nFeatures);
        if (!s) return s;
    }
    else if (outputMatrixType == correlationMatrix)
    {
        /* Check correlation matrix */
        s |= checkNumericTable(get(correlation).get(), correlationStr(), unexpectedLayout, 0, nFeatures, nFeatures);
        if (!s) return s;
    }

    /* Check mean vector */
    s |= checkNumericTable(get(mean).get(), meanStr(), 0, 0, nFeatures, 1);
    return s;
}

} //namespace interface1

} //namespace covariance
} // namespace algorithms
} // namespace daal
