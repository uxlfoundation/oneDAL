/* file: svm_quality_metric_set_types.h */
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
//  Interface for the SVM algorithm quality metrics
//--
*/

#ifndef __SVM_QUALITY_METRIC_SET_TYPES_H__
#define __SVM_QUALITY_METRIC_SET_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm_quality_metric_set_types.h"
#include "algorithms/classifier/binary_confusion_matrix_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
/**
 * @defgroup svm_quality_metric_set Quality Metrics
 * \copydoc daal::algorithms::svm::quality_metric_set
 * @ingroup svm
 * @{
 */
namespace quality_metric_set
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVM__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * Available identifiers of the quality metrics available for the model trained with the SVM algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 *        specialized for using with the SVM training algorithm
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
    classifier::quality_metric::binary_confusion_matrix::ResultPtr getResult(QualityMetricId id) const;
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 *        specialized for using with the SVM training algorithm
 */
class DAAL_EXPORT InputDataCollection : public algorithms::quality_metric_set::InputDataCollection
{
public:
    InputDataCollection() {}
    virtual ~InputDataCollection() {}

    /**
     * Returns the input object of the quality metrics algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    classifier::quality_metric::binary_confusion_matrix::InputPtr getInput(QualityMetricId id) const;
};
typedef services::SharedPtr<InputDataCollection> InputDataCollectionPtr;
} // namespace interface1
using interface1::ResultCollection;
using interface1::ResultCollectionPtr;
using interface1::InputDataCollection;
using interface1::InputDataCollectionPtr;

} // namespace quality_metric_set
/** @} */
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif // __SVM_QUALITY_METRIC_SET_TYPES_H__
