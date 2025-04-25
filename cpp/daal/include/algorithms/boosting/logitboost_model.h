/* file: logitboost_model.h */
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

#ifndef __LOGIT_BOOST_MODEL_H__
#define __LOGIT_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/regression/regression_model.h"
#include "algorithms/regression/regression_training_batch.h"
#include "algorithms/regression/regression_predict.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the LogitBoost classification algorithm
 */
namespace logitboost
{
/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup logitboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOGITBOOST__PARAMETER"></a>
 * \brief LogitBoost algorithm parameters
 *
 * \snippet boosting/logitboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    /** Default constructor */
    Parameter();

    /**
     * Constructs LogitBoost parameter structure
     * \param[in] wlTrainForParameter       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter     Pointer to the prediction algorithm of the weak learner
     * \param[in] acc                       Accuracy of the LogitBoost training algorithm
     * \param[in] maxIter                   Maximal number of terms in additive regression
     * \param[in] nC                        Number of classes in the training data set
     * \param[in] wThr                      Threshold to avoid degenerate cases when calculating weights W
     * \param[in] zThr                      Threshold to avoid degenerate cases when calculating responses Z
     */
    Parameter(const services::SharedPtr<regression::training::Batch> & wlTrainForParameter,
              const services::SharedPtr<regression::prediction::Batch> & wlPredictForParameter, double acc = 0.0, size_t maxIter = 10, size_t nC = 0,
              double wThr = 1e-10, double zThr = 1e-10);

    services::SharedPtr<regression::training::Batch> weakLearnerTraining;     /*!< The algorithm for weak learner model training */
    services::SharedPtr<regression::prediction::Batch> weakLearnerPrediction; /*!< The algorithm for prediction based on a weak learner model */
    double accuracyThreshold;                                                 /*!< Accuracy of the LogitBoost training algorithm */
    size_t maxIterations;                                                     /*!< Maximal number of terms in additive regression */
    double weightsDegenerateCasesThreshold;                                   /*!< Threshold to avoid degenerate cases when  calculating weights W */
    double responsesDegenerateCasesThreshold; /*!< Threshold to avoid degenerate cases when  calculating responses Z */
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the logitboost::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface2::Batch "training::Batch" class
 *      - \ref prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
     * Constructs the LogitBoost model
     * \tparam modelFPType  Data type to store LogitBoost model data, double or float
     * \param[in] nFeatures Number of features in the dataset
     * \param[in] par       Pointer to the parameter structure of the LogitBoost algorithm
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, const Parameter * par, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(size_t nFeatures = 0) : _nFeatures(nFeatures), _models(new data_management::DataCollection()), _nIterations(0) {}

    /**
     * Constructs the LogitBoost model
     * \param[in]  nFeatures Number of features in the dataset
     * \param[in]  par       Pointer to the parameter structure of the LogitBoost algorithm
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(size_t nFeatures, const Parameter * par, services::Status * stat = NULL);

    virtual ~Model() {}

    /**
     *  Returns the number of weak learners constructed during training of the LogitBoost algorithm
     *  \return The number of weak learners
     */
    size_t getNumberOfWeakLearners() const;

    /**
     *  Returns weak learner model constructed during training of the LogitBoost algorithm
     *  \param[in] idx  Index of the model in the collection
     *  \return Weak Learner model corresponding to the index idx
     */
    regression::ModelPtr getWeakLearnerModel(size_t idx) const;

    /**
     *  Add weak learner model into the LogitBoost model
     *  \param[in] model Weak learner model to add into collection
     */
    void addWeakLearnerModel(regression::ModelPtr model);

    /**
     *  Clears the collecion of weak learners
     */
    void clearWeakLearnerModels();

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return _nFeatures; }

    /**
     * Sets the number of iterations for the algorithm
     * @param nIterations   Number of iterations
     */
    void setIterations(size_t nIterations);

    /**
     * Returns the number of iterations done by the training algorithm
     * \return The number of iterations done by the training algorithm
     */
    size_t getIterations() const;

protected:
    size_t _nFeatures;
    data_management::DataCollectionPtr _models;
    size_t _nIterations;

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st;
        DAAL_CHECK_STATUS(st, (classifier::Model::serialImpl<Archive, onDeserialize>(arch)));
        arch->set(_nFeatures);
        arch->setSharedPtrObj(_models);
        arch->set(_nIterations);

        return st;
    }

    Model(size_t nFeatures, const Parameter * par, services::Status & st);
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface2
using interface2::Parameter;
using interface2::Model;
using interface2::ModelPtr;

} // namespace logitboost
} // namespace algorithms
} // namespace daal
#endif
