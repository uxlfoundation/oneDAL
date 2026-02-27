/* file: logitboost_model.cpp */
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
//  Implementation of class defining LogitBoost model.
//--
*/

#include "algorithms/boosting/logitboost_model.h"
#include "algorithms/stump/stump_regression_training_batch.h"
#include "algorithms/stump/stump_regression_predict.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace interface2
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_LOGITBOOST_MODEL_ID);
/** Default constructor */
Parameter::Parameter()
    : classifier::Parameter(0),
      weakLearnerTraining(new stump::regression::training::Batch<>),
      weakLearnerPrediction(new stump::regression::prediction::Batch<>),
      accuracyThreshold(0.0),
      maxIterations(10),
      weightsDegenerateCasesThreshold(1e-10),
      responsesDegenerateCasesThreshold(1e-10)
{}

/**
 * Constructs LogitBoost parameter structure
 * \param[in] wlTrain       Pointer to the training algorithm of the weak learner
 * \param[in] wlPredict     Pointer to the prediction algorithm of the weak learner
 * \param[in] acc           Accuracy of the LogitBoost training algorithm
 * \param[in] maxIter       Maximal number of terms in additive regression
 * \param[in] nC            Number of classes in the training data set
 * \param[in] wThr          Threshold to avoid degenerate cases when calculating weights W
 * \param[in] zThr          Threshold to avoid degenerate cases when calculating responses Z
 */
Parameter::Parameter(const SharedPtr<regression::training::Batch> & wlTrain, const SharedPtr<regression::prediction::Batch> & wlPredict, double acc,
                     size_t maxIter, size_t nC, double wThr, double zThr)
    : classifier::Parameter(nC),
      weakLearnerTraining(wlTrain),
      weakLearnerPrediction(wlPredict),
      accuracyThreshold(acc),
      maxIterations(maxIter),
      weightsDegenerateCasesThreshold(wThr),
      responsesDegenerateCasesThreshold(zThr)
{}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_EX(accuracyThreshold >= 0 && accuracyThreshold < 1, ErrorIncorrectParameter, ParameterName, accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations > 0, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX(nClasses >= 2, ErrorIncorrectParameter, ParameterName, nClassesStr());
    DAAL_CHECK_EX(weightsDegenerateCasesThreshold > 0, ErrorIncorrectParameter, ParameterName, weightsDegenerateCasesThresholdStr());
    DAAL_CHECK_EX(responsesDegenerateCasesThreshold > 0, ErrorIncorrectParameter, ParameterName, responsesDegenerateCasesThresholdStr());
    return s;
}

Model::Model(size_t nFeatures, const Parameter * par, services::Status & st)
    : _nFeatures(nFeatures), _models(new data_management::DataCollection()), _nIterations(par->maxIterations)
{
    if (!_models)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
}

/**
 * Constructs the LogitBoost model
 * \param[in]  nFeatures Number of features in the dataset
 * \param[in]  par       Pointer to the parameter structure of the LogitBoost algorithm
 * \param[out] stat      Status of the model construction
 */
ModelPtr Model::create(size_t nFeatures, const Parameter * par, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, par);
}

/**
 * Sets the number of iterations for the algorithm
 * @param nIterations   Number of iterations
 */
void Model::setIterations(size_t nIterations)
{
    _nIterations = nIterations;
}

/**
 * Returns the number of iterations done by the training algorithm
 * \return The number of iterations done by the training algorithm
 */
size_t Model::getIterations() const
{
    return _nIterations;
}

size_t Model::getNumberOfWeakLearners() const
{
    return _models->size();
}

regression::ModelPtr Model::getWeakLearnerModel(size_t idx) const
{
    if (idx < _models->size())
    {
        return staticPointerCast<regression::Model, SerializationIface>((*_models)[idx]);
    }
    return regression::ModelPtr();
}

void Model::addWeakLearnerModel(regression::ModelPtr model)
{
    (*_models) << model;
}

void Model::clearWeakLearnerModels()
{
    _models->clear();
}

} // namespace interface2
} // namespace logitboost
} // namespace algorithms
} // namespace daal
