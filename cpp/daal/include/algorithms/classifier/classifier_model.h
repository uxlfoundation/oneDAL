/* file: classifier_model.h */
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
//  Implementation of the class defining the model of the classification  algorithm
//--
*/

#ifndef __CLASSIFIER_MODEL_H__
#define __CLASSIFIER_MODEL_H__

#include "algorithms/algorithm.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup classification Classification
 * \brief Contains classes for work with the classification algorithms
 * @ingroup training_and_prediction
 */
namespace classifier
{
/**
* <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__RESULTOCOMPUTETID"></a>
* Available identifiers of optional results of the classifier algorithm
* @ingroup training_and_prediction
*/
enum ResultToComputeId
{
    none                         = 0x00000000ULL,
    computeClassLabels           = 0x00000001ULL, /*!< Numeric table of size n x 1 with the predicted labels >*/
    computeClassProbabilities    = 0x00000002ULL, /*!< Numeric table of size n x p with the predicted class probabilities for each observation >*/
    computeClassLogProbabilities = 0x00000004ULL  /*!< Numeric table of size n x p with the predicted class probabilities for each observation >*/
};

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup classifier
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__CLASSIFIER__PARAMETER"></a>
 * \brief Base class for the parameters of the classification algorithm
 *
 * \snippet classifier/classifier_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nClasses = 2);

    size_t nClasses;               /*!< Number of classes */
    DAAL_UINT64 resultsToEvaluate; /*!< 64 bit integer flag that indicates the results to compute */
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
/** @} */
} // namespace interface2
using interface2::Parameter;

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup classifier
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__MODEL"></a>
 * \brief Base class for the model of the classification algorithm
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    DAAL_CAST_OPERATOR(Model)

    Model()                                = default;
    Model(const Model & other)             = default;
    Model & operator=(const Model & other) = default;

    virtual ~Model() DAAL_C11_OVERRIDE {}

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \DAAL_DEPRECATED_USE{ Model::getNumberOfFeatures }
     *  \return Number of features in the dataset was used on the training stage
     */
    virtual size_t getNFeatures() const { return getNumberOfFeatures(); }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    virtual size_t getNumberOfFeatures() const = 0;

    /**
     *  Sets the number of features in the dataset was used on the training stage
     *  \DAAL_DEPRECATED
     *  \param[in]  nFeatures  Number of features in the dataset was used on the training stage
     */
    virtual void setNFeatures(size_t /*nFeatures*/) {}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
