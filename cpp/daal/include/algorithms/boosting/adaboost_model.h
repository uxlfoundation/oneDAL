/* file: adaboost_model.h */
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
//  Implementation of class defining Ada Boost model.
//--
*/

#ifndef __ADA_BOOST_MODEL_H__
#define __ADA_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/classifier/classifier_predict.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the AdaBoost classification algorithm
 */
namespace adaboost
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute
 */
enum ResultToComputeId
{
    computeWeakLearnersErrors = 0x00000001ULL
};
/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup adaboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__ADABOOST__PARAMETER"></a>
 * \brief AdaBoost algorithm parameters
 *
 * \snippet boosting/adaboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    /**
     * Default contructor
     * @param nClasses The number of classes
     */
    Parameter(size_t nClasses = 2);

    /**
     * Constructs the AdaBoost parameter structure
     * \param[in] wlTrainForParameter    Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter  Pointer to the prediction algorithm of the weak learner
     * \param[in] acc                    Accuracy of the AdaBoost training algorithm
     * \param[in] maxIter                Maximal number of iterations of the AdaBoost training algorithm
     * \param[in] learnRate              Multiplier for each classifier to shrink its contribution
     * \param[in] resToCompute           The 64-bit integer flag that specifies which extra characteristics of the AdaBoost compute from \ref ResultToComputeId
     * \param[in] nCl                    Number of classes
     */
    Parameter(services::SharedPtr<classifier::training::Batch> wlTrainForParameter,
              services::SharedPtr<classifier::prediction::Batch> wlPredictForParameter, double acc = 0.0, size_t maxIter = 10, double learnRate = 1.0,
              DAAL_UINT64 resToCompute = computeWeakLearnersErrors, size_t nCl = 2);

    services::SharedPtr<classifier::training::Batch> weakLearnerTraining;     /*!< The algorithm for weak learner model training */
    services::SharedPtr<classifier::prediction::Batch> weakLearnerPrediction; /*!< The algorithm for prediction based on a weak learner model */
    double accuracyThreshold;                                                 /*!< Accuracy of the AdaBoost training algorithm */
    size_t maxIterations;                                                     /*!< Maximal number of iterations of the AdaBoost training algorithm */
    double learningRate;                                                      /*!< Multiplier for each classifier to shrink its contribution */
    DAAL_UINT64 resultsToCompute;                                             /*!< 64 bit integer flag that indicates the results to compute */
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the adaboost::training::Batch algorithm.
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
     * Constructs the AdaBoost model
     * \tparam modelFPType  Data type to store AdaBoost model data, double or float
     * \param[in] nFeatures Number of features in the dataset
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(size_t nFeatures = 0) : _nFeatures(nFeatures), _models(new data_management::DataCollection()), _alpha() {}

    /**
     * Constructs the AdaBoost model
     * \tparam modelFPType   Data type to store AdaBoost model data, double or float
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nFeatures, services::Status * stat = NULL);

    virtual ~Model() {}

    /**
     *  Returns the number of weak learners constructed during training of the AdaBoost algorithm
     *  \return The number of weak learners
     */
    size_t getNumberOfWeakLearners() const;

    /**
     *  Returns weak learner model constructed during training of the AdaBoost algorithm
     *  \param[in] idx  Index of the model in the collection
     *  \return Weak Learner model corresponding to the index idx
     */
    classifier::ModelPtr getWeakLearnerModel(size_t idx) const;

    /**
     *  Add weak learner model into the AdaBoost model
     *  \param[in] model Weak learner model to add into collection
     */
    void addWeakLearnerModel(classifier::ModelPtr model);

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
     *  Returns a pointer to the array of weights of weak learners constructed
     *  during training of the AdaBoost algorithm.
     *  The size of the array equals the number of weak learners
     *  \return Array of weights of weak learners.
     */
    data_management::NumericTablePtr getAlpha() const;

protected:
    size_t _nFeatures;
    data_management::DataCollectionPtr _models;
    data_management::NumericTablePtr _alpha; /* Boosting coefficients table */

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st;
        DAAL_CHECK_STATUS(st, (classifier::Model::serialImpl<Archive, onDeserialize>(arch)));
        arch->set(_nFeatures);
        arch->setSharedPtrObj(_models);
        arch->setSharedPtrObj(_alpha);

        return st;
    }

    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy, services::Status & st);
}; // class Model
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface2
using interface2::Parameter;
using interface2::Model;
using interface2::ModelPtr;

} // namespace adaboost
} // namespace algorithms
} // namespace daal
#endif
