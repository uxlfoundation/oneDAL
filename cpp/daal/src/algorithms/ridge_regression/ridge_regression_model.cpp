/* file: ridge_regression_model.cpp */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#include "algorithms/ridge_regression/ridge_regression_model.h"
#include "algorithms/ridge_regression/ridge_regression_ne_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace interface1
{
TrainParameter::TrainParameter() : Parameter(), ridgeParameters(new HomogenNumericTable<double>(1, 1, NumericTableIface::doAllocate, 1.0)) {};

services::Status TrainParameter::check() const
{
    return checkNumericTable(ridgeParameters.get(), ridgeParametersStr(), packed_mask, 0, 0, 1);
}

} // namespace interface1

Status checkModel(ridge_regression::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses, int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, linear_model::checkModel(model, par, nBeta, nResponses));

    size_t dimWithoutBeta = (model->getInterceptFlag() ? nBeta : nBeta - 1);

    ridge_regression::ModelNormEq * modelNormEq = dynamic_cast<ridge_regression::ModelNormEq *>(model);
    DAAL_CHECK(modelNormEq, ErrorNullModel);

    DAAL_CHECK_STATUS(s, checkNumericTable(modelNormEq->getXTXTable().get(), XTXTableStr(), 0, 0, dimWithoutBeta, dimWithoutBeta));
    return checkNumericTable(modelNormEq->getXTYTable().get(), XTYTableStr(), 0, 0, dimWithoutBeta, nResponses);
}
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
