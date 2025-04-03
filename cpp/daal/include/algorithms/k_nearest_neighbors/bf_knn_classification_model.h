/* file: bf_knn_classification_model.h */
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) classification model
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_MODEL_H__
#define __BF_KNN_CLASSIFICATION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"
#include "algorithms/engines/mcg59/mcg59.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup bf_knn_classification k-Nearest Neighbors
 * \copydoc daal::algorithms::bf_knn_classification
 * @ingroup classification
 * @{
 */

/**
 * \brief Contains classes for BF kNN algorithm
 */
namespace bf_knn_classification
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__DATAUSEINMODEL"></a>
 * \brief The option to enable/disable an usage of the input dataset in kNN model
 */
enum DataUseInModel
{
    doNotUse = 0, /*!< The input data and labels will not be the component of the trained kNN model */
    doUse    = 1  /*!< The input data and labels will be the component of the trained kNN model */
};
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute
 */
enum ResultToComputeId
{
    computeIndicesOfNeighbors = 0x00000001ULL, /*!< The flag to compute indices of nearest neighbors */
    computeDistances          = 0x00000002ULL  /*!< The flag to compute distances to nearest neighbors */
};
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__RESULTTOCOMPUTEID"></a>
 * \brief Weight function used in prediction voting
 */
enum VoteWeights
{
    voteUniform  = 0, /*!< Uniform weights for neighbors for prediction voting. All neighbors are weighted equally */
    voteDistance = 1  /*!< Weight neighbors by the inverse of their distance. Closer neighbors of a query point will have a greater influence
                           than neighbors that are further away */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__BF_KNN_CLASSIFICATION__PARAMETER"></a>
 * \brief BF kNN algorithm parameters
 *
 * \snippet k_nearest_neighbors/bf_knn_classification_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    /**
     *  Parameter constructor
     *  \param[in] nClasses             Number of classes
     *  \param[in] nNeighbors           Number of neighbors
     *  \param[in] dataUse              The option to enable/disable an usage of the input dataset in kNN model
     *  \param[in] resToCompute         64 bit integer flag that indicates the results to compute
     *  \param[in] resToEvaluate        64 bit integer flag that indicates the results to evaluate
     *  \param[in] vote                 The option to select voting method
     */
    Parameter(size_t nClasses = 2, size_t nNeighbors = 1, DataUseInModel dataUse = doNotUse, DAAL_UINT64 resToCompute = 0,
              DAAL_UINT64 resToEvaluate = daal::algorithms::classifier::computeClassLabels, VoteWeights vote = voteUniform)
        : daal::algorithms::classifier::Parameter(nClasses),
          k(nNeighbors),
          dataUseInModel(dataUse),
          resultsToCompute(resToCompute),
          voteWeights(vote),
          engine(engines::mcg59::Batch<>::create())
    {
        this->resultsToEvaluate = resToEvaluate;
    }

    /**
     *  Parameter copy constructor
     *  \param[in] other             Object to copy
     */
    Parameter(const Parameter & other)
        : daal::algorithms::classifier::Parameter(other.nClasses),
          k(other.k),
          dataUseInModel(other.dataUseInModel),
          resultsToCompute(other.resultsToCompute),
          voteWeights(other.voteWeights),
          engine(other.engine->clone())
    {
        this->resultsToEvaluate = other.resultsToEvaluate;
    }

    /**
     *  Parameter copy constructor
     *  \param[in] other             Object to copy
     */
    Parameter & operator=(const Parameter & other)
    {
        if (this != &other)
        {
            daal::algorithms::classifier::Parameter::operator=(other);
            k                       = other.k;
            dataUseInModel          = other.dataUseInModel;
            engine                  = other.engine->clone();
            voteWeights             = other.voteWeights;
            resultsToCompute        = other.resultsToCompute;
            this->resultsToEvaluate = other.resultsToEvaluate;
        }
        return *this;
    }

    virtual ~Parameter() {}
    /**
     * Checks a parameter of the BF kNN algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    size_t k;                      /*!< Number of neighbors */
    DataUseInModel dataUseInModel; /*!< The option to enable/disable an usage of the input dataset in kNN model */
    DAAL_UINT64 resultsToCompute;  /*!< 64 bit integer flag that indicates the results to compute */
    VoteWeights voteWeights;       /*!< Weight function used in prediction */
    engines::EnginePtr engine;     /*!< Engine for random choosing elements from training dataset */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__MODEL"></a>
 * \brief %Base class for models trained with the BF kNN algorithm
 *
 * \tparam modelFPType  Data type to store BF kNN model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::classifier::Model
{
public:
    DECLARE_MODEL_IFACE(Model, classifier::Model);
    /**
     * Constructs the model trained with the BF kNN algorithm
     * \param[in] nFeatures Number of features in the dataset
     */
    Model(size_t nFeatures = 0);

    virtual ~Model() DAAL_C11_OVERRIDE;

    class ModelImpl;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl * impl() const { return _impl; }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl * impl() { return _impl; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

protected:
    Model(size_t nFeatures, services::Status & st);

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    ModelImpl * _impl; /*!< Model implementation */

    Model(const Model &);
    Model & operator=(const Model &);
};
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace bf_knn_classification

/** @} */
} // namespace algorithms
} // namespace daal

#endif
