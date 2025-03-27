/* file: linear_model_model.cpp */
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
//  Implementation of the class defining the regression model
//--
*/

#include "src/algorithms/linear_model/linear_model_model_impl.h"
#include "algorithms/linear_model/linear_model_model.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
using namespace daal::services;
using namespace daal::data_management;

namespace interface1
{
Parameter::Parameter() : algorithms::Parameter(), interceptFlag(true) {}
Parameter::Parameter(const Parameter & other)             = default;
Parameter & Parameter::operator=(const Parameter & other) = default;
} // namespace interface1

namespace internal
{
ModelInternal::ModelInternal() : _interceptFlag(true), _beta() {}

ModelInternal::ModelInternal(const NumericTablePtr & beta, const linear_model::Parameter & par) : _beta(beta), _interceptFlag(par.interceptFlag) {}

Status ModelInternal::initialize()
{
    const size_t nRows = _beta->getNumberOfRows();
    daal::internal::WriteOnlyRows<float, DAAL_BASE_CPU> betaRows(*_beta, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(betaRows);
    float * betaArray     = betaRows.get();
    const size_t betaSize = _beta->getNumberOfColumns() * nRows;
    for (size_t i = 0; i < betaSize; i++)
    {
        betaArray[i] = 0.0f;
    }
    return Status();
}

size_t ModelInternal::getNumberOfBetas() const
{
    return _beta->getNumberOfColumns();
}

size_t ModelInternal::getNumberOfFeatures() const
{
    return getNumberOfBetas() - 1;
}

size_t ModelInternal::getNumberOfResponses() const
{
    return _beta->getNumberOfRows();
}

bool ModelInternal::getInterceptFlag() const
{
    return _interceptFlag;
}

NumericTablePtr ModelInternal::getBeta()
{
    return _beta;
}

Status ModelInternal::setToZero(NumericTable & table)
{
    const size_t nRows = table.getNumberOfRows();
    daal::internal::WriteOnlyRows<float, DAAL_BASE_CPU> tableRows(table, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(tableRows);
    float * tableArray = tableRows.get();

    const size_t nCols = table.getNumberOfColumns();

    for (size_t i = 0; i < nCols * nRows; i++)
    {
        tableArray[i] = 0.0f;
    }

    return Status();
}

} // namespace internal

Status checkModel(linear_model::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses)
{
    DAAL_CHECK(model, ErrorNullModel);

    const Parameter & parameter = static_cast<const Parameter &>(par);
    DAAL_CHECK_EX(model->getInterceptFlag() == parameter.interceptFlag, ErrorIncorrectParameter, ParameterName, interceptFlagStr());

    return checkNumericTable(model->getBeta().get(), betaStr(), 0, 0, nBeta, nResponses);
}
} // namespace linear_model
} // namespace algorithms
} // namespace daal
