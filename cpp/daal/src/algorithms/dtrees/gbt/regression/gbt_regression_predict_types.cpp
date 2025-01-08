/* file: gbt_regression_predict_types.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_REGRESSION_PREDICTION_RESULT_ID);

/** Default constructor */
Input::Input() : algorithms::regression::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(NumericTableInputId id) const
{
    return algorithms::regression::prediction::Input::get(algorithms::regression::prediction::NumericTableInputId(id));
}

/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
gbt::regression::ModelPtr Input::get(ModelInputId id) const
{
    return staticPointerCast<gbt::regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(NumericTableInputId id, const NumericTablePtr & value)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::NumericTableInputId(id), value);
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const gbt::regression::ModelPtr & value)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::ModelInputId(id), value);
}

/**
 * Checks an input object for making gradient boosted trees model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::prediction::Input::check(parameter, method));

    ModelPtr m = get(prediction::model);
    const daal::algorithms::gbt::regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::regression::internal::ModelImpl *>(m.get());
    DAAL_ASSERT(pModel);
    DAAL_CHECK(pModel->getNumberOfTrees(), services::ErrorNullModel);
    auto maxNIterations = pModel->getNumberOfTrees();

    const gbt::regression::prediction::interface1::Parameter * pPrm =
        static_cast<const gbt::regression::prediction::interface1::Parameter *>(parameter);
    size_t nIterations = pPrm->nIterations;

    DAAL_CHECK((nIterations == 0) || (nIterations <= maxNIterations), services::ErrorGbtPredictIncorrectNumberOfIterations);
    const bool predictContribs     = pPrm->resultsToCompute & shapContributions;
    const bool predictInteractions = pPrm->resultsToCompute & shapInteractions;
    DAAL_CHECK(!(predictContribs && predictInteractions), services::ErrorGbtPredictShapOptions);
    return s;
}

Result::Result() : algorithms::regression::prediction::Result(lastResultId + 1) {};

/**
 * Returns the result of gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return algorithms::regression::prediction::Result::get(algorithms::regression::prediction::ResultId(id));
}

/**
 * Sets the result of gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    algorithms::regression::prediction::Result::set(algorithms::regression::prediction::ResultId(id), value);
}

/**
 * Checks the result of gradient boosted trees model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    using algorithms::regression::prediction::Result;

    Status s;
    DAAL_CHECK_STATUS(s, Result::check(input, par, method));
    const auto inputCast                              = static_cast<const prediction::Input *>(input);
    const prediction::Parameter * regressionParameter = static_cast<const prediction::Parameter *>(par);
    size_t expectedNColumns                           = 1;
    if (regressionParameter->resultsToCompute & shapContributions)
    {
        const size_t nColumns = inputCast->get(data)->getNumberOfColumns();
        expectedNColumns      = nColumns + 1;
    }
    else if (regressionParameter->resultsToCompute & shapInteractions)
    {
        const size_t nColumns = inputCast->get(data)->getNumberOfColumns();
        expectedNColumns      = (nColumns + 1) * (nColumns + 1);
    }
    DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == expectedNColumns, ErrorIncorrectNumberOfColumns, ArgumentName, predictionStr());
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
