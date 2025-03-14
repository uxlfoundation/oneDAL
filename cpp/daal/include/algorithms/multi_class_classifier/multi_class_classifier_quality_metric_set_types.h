/* file: multi_class_classifier_quality_metric_set_types.h */
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
//  Interface for the multi-class algorithm quality metrics
//--
*/

#ifndef __MULTI_CLASS_QUALITY_METRIC_SET_TYPES_H__
#define __MULTI_CLASS_QUALITY_METRIC_SET_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm_quality_metric_set_types.h"
#include "algorithms/classifier/multiclass_confusion_matrix_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * @defgroup multi_class_classifier_quality_metric_set Quality Metrics
 * \copydoc daal::algorithms::multi_class_classifier::quality_metric_set
 * @ingroup multi_class_classifier
 * @{
 */
/**
 * \brief Contains classes for checking the quality of the model trained with the multi-class classifier algorithm
 */
namespace quality_metric_set
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * Available identifiers of the quality metrics for the model trained with the multi-class classifier algorithm
 */
enum QualityMetricId
{
    confusionMatrix, /*!< Confusion matrix */
    lastQualityMetricId = confusionMatrix
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
* <a name="DAAL-STRUCT-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__PARAMETER"></a>
* \brief Parameters for the multi-class classifier compute() method
*
* \snippet multi_class_classifier/multi_class_classifier_quality_metric_set_types.h Parameter source code
*/
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nClasses = 2);
    virtual ~Parameter() {}

    size_t nClasses; /*!< Number of classes */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 *        specialized for using with the multi-class classifier training algorithm
 */
class DAAL_EXPORT ResultCollection : public algorithms::quality_metric_set::ResultCollection
{
public:
    ResultCollection() {}
    virtual ~ResultCollection() {}

    /**
     * Returns the result of the quality metrics algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    classifier::quality_metric::multiclass_confusion_matrix::ResultPtr getResult(QualityMetricId id) const;
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 *        specialized for using with the multi-class classifier training algorithm
 */
class DAAL_EXPORT InputDataCollection : public algorithms::quality_metric_set::InputDataCollection
{
public:
    InputDataCollection() {}
    virtual ~InputDataCollection() {}

    /**
     * Returns the input object for the quality metrics algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    classifier::quality_metric::multiclass_confusion_matrix::InputPtr getInput(QualityMetricId id) const;
};
typedef services::SharedPtr<InputDataCollection> InputDataCollectionPtr;
} // namespace interface1
using interface1::Parameter;
using interface1::ResultCollection;
using interface1::ResultCollectionPtr;
using interface1::InputDataCollection;
using interface1::InputDataCollectionPtr;

} // namespace quality_metric_set
/** @} */
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif // __MULTI_CLASS_QUALITY_METRIC_SET_TYPES_H__
