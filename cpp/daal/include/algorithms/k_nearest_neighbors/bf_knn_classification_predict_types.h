/* file: bf_knn_classification_predict_types.h */
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
//  Implementation of the K-Nearest Neighbors (kNN) algorithm interface
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_TYPES_H__
#define __BF_KNN_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the BF kNN algorithm
 */
namespace bf_knn_classification
{
/**
 * @defgroup bf_knn_classification_prediction Prediction
 * \copydoc daal::algorithms::bf_knn_classification::prediction
 * @ingroup bf_knn_classification
 * @{
 */
/**
 * \brief Contains a class for making BF kNN model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__METHOD"></a>
 * \brief Available methods for making BF kNN model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making brute-force kNN model-based prediction
 */
enum ResultId
{
    prediction   = classifier::prediction::prediction,       /*!< Prediction results */
    indices      = classifier::prediction::lastResultId + 1, /*!< Indices of nearest neighbors */
    distances    = classifier::prediction::lastResultId + 2, /*!< Distances to nearest neighbors */
    lastResultId = distances
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making BF kNN model-based prediction
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    /** Default constructor */
    Input();

    using super::get;
    using super::set;

    /**
     * Returns the input Model object in the prediction stage of the BF kNN algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    bf_knn_classification::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the BF kNN algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(classifier::prediction::ModelInputId id, const bf_knn_classification::interface1::ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of brute-force kNN model-based prediction
 */
class DAAL_EXPORT Result : public classifier::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of brute-force kNN model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of brute-force kNN model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory for storing prediction results of brute-force kNN algorithm
     * \tparam  algorithmFPType     Data type for storing prediction results
     * \param[in] input     Pointer to the input objects of the classification algorithm
     * \param[in] parameter Pointer to the parameters of the classification algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method);

    /**
     * Checks the correctness of prediction results of brute-force kNN algorithm
     * \param[in] input     Pointer to the the input object
     * \param[in] parameter Pointer to the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using classifier::prediction::Result::check;

    /** \private */
    services::Status checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return classifier::prediction::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;

} // namespace interface1

using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace prediction
/** @} */
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
