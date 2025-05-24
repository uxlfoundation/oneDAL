/* file: gbt_classification_predict_types.cpp */
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

#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"
#include "algorithms/classifier/classifier_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace interface1
{
Input::Input()                                = default;
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return algorithms::classifier::prediction::Input::get(id);
}

/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
gbt::classification::ModelPtr Input::get(ModelInputId id) const
{
    return staticPointerCast<gbt::classification::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr & value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const gbt::classification::ModelPtr & value)
{
    algorithms::classifier::prediction::Input::set(algorithms::classifier::prediction::ModelInputId(id), value);
}

/**
 * Checks an input object for making gradient boosted trees model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::classifier::prediction::Input::check(parameter, method));
    classifier::ModelPtr m = get(prediction::model);
    const auto * pModel    = static_cast<const classification::internal::ModelImpl *>(m.get());
    DAAL_ASSERT(pModel);
    DAAL_CHECK(pModel->getNumberOfTrees(), services::ErrorNullModel);

    size_t nClasses = 0, nIterations = 0;

    const auto * pPrm = dynamic_cast<const gbt::classification::prediction::Parameter *>(parameter);
    if (pPrm)
    {
        nClasses    = pPrm->nClasses;
        nIterations = pPrm->nIterations;
    }
    else
    {
        return services::ErrorNullParameterNotSupported;
    }

    const bool predictContribs     = pPrm->resultsToCompute & shapContributions;
    const bool predictInteractions = pPrm->resultsToCompute & shapInteractions;
    DAAL_CHECK(!(predictContribs && predictInteractions), services::ErrorGbtPredictShapOptions);
    DAAL_CHECK(!(nClasses > 2 && (predictContribs || predictInteractions)), services::ErrorGbtPredictShapMulticlassNotSupported);

    auto maxNIterations = pModel->getNumberOfTrees();
    if (nClasses > 2) maxNIterations /= nClasses;
    DAAL_CHECK((nClasses < 3) || (pModel->getNumberOfTrees() % nClasses == 0), services::ErrorGbtIncorrectNumberOfTrees);
    DAAL_CHECK((nIterations == 0) || (nIterations <= maxNIterations), services::ErrorGbtPredictIncorrectNumberOfIterations);

    return s;
}

} // namespace interface1

namespace interface2
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_CLASSIFICATION_PREDICTION_RESULT_ID);

Result::Result() : algorithms::classifier::prediction::Result(lastResultId + 1) {};

/**
 * Returns the result of gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return algorithms::classifier::prediction::Result::get(algorithms::classifier::prediction::ResultId(id));
}

/**
 * Sets the result of gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr & value)
{
    algorithms::classifier::prediction::Result::set(algorithms::classifier::prediction::ResultId(id), value);
}

/**
 * Checks the result of gradient boosted trees model-based prediction
 * \param[in] input   %Input object
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    using algorithms::classifier::computeClassLabels;
    using algorithms::classifier::computeClassProbabilities;
    using algorithms::classifier::computeClassLogProbabilities;
    using algorithms::classifier::prediction::data;
    using algorithms::classifier::prediction::Result;

    const prediction::Parameter * classificationParameter = static_cast<const prediction::Parameter *>(par);
    Status s;

    const auto predictionInput = static_cast<const prediction::Input *>(input);
    DAAL_CHECK(predictionInput->get(data).get(), services::ErrorNullInputNumericTable);
    const size_t nRows = predictionInput->get(data)->getNumberOfRows();

    classifier::ModelPtr m = predictionInput->get(prediction::model);
    DAAL_CHECK(m.get(), services::ErrorNullModel);

    if (classificationParameter->resultsToEvaluate & computeClassLabels)
    {
        size_t expectedNColumns = 1;
        if (classificationParameter->resultsToCompute & shapContributions)
        {
            const size_t nColumns = predictionInput->get(data)->getNumberOfColumns();
            expectedNColumns      = nColumns + 1;
        }
        else if (classificationParameter->resultsToCompute & shapInteractions)
        {
            const size_t nColumns = predictionInput->get(data)->getNumberOfColumns();
            expectedNColumns      = (nColumns + 1) * (nColumns + 1);
        }
        DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == expectedNColumns, ErrorIncorrectNumberOfColumns, ArgumentName, predictionStr());
        DAAL_CHECK_STATUS(
            s, data_management::checkNumericTable(get(prediction).get(), predictionStr(), data_management::packed_mask, 0, expectedNColumns, nRows));
    }
    if (classificationParameter->resultsToEvaluate & computeClassProbabilities)
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(probabilities).get(), probabilitiesStr(), data_management::packed_mask, 0,
                                                                classificationParameter->nClasses, nRows));
    if (classificationParameter->resultsToEvaluate & computeClassLogProbabilities)
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(logProbabilities).get(), logProbabilitiesStr(), data_management::packed_mask, 0,
                                                                classificationParameter->nClasses, nRows));

    return s;
}

} // namespace interface2

} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
