/* file: pca_quality_metric_set_types.h */
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
//  Interface for the pca algorithm quality metrics
//--
*/

#ifndef __PCA_QUALITY_METRIC_SET_TYPES_H__
#define __PCA_QUALITY_METRIC_SET_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "algorithms/algorithm_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
 * @defgroup pca_quality_metric_set Quality Metrics
 * \copydoc daal::algorithms::pca::quality_metric_set
 * @ingroup pca
 * @{
 */
namespace quality_metric_set
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * Available identifiers of the quality metrics of pca algorithm
 */
enum QualityMetricId
{
    explainedVariancesMetrics, /*!< Explained and noise variances metrics*/
    lastQualityMetricId = explainedVariancesMetrics
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
* <a name="DAAL-STRUCT-ALGORITHMS__PCA__QUALITY_METRIC_SET__PARAMETER"></a>
* \brief Parameters for the quality metrics set compute() method
*
* \snippet pca/pca_quality_metric_set_types.h Parameter source code
*/
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    DAAL_EXPORT Parameter(size_t nComponents = 0, size_t nFeatures = 0);

    virtual ~Parameter() {}

    size_t nComponents; /*!< Number of principal components to compute metrics for */
    size_t nFeatures;   /*!< Number of features in dataset used as input in PCA */

    /**
    * Checks the correctness of the parameter
    *
     * \return Status of computations
     */
    DAAL_EXPORT virtual services::Status check() const;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 *        specialized for using with the pca algorithm
 */
class ResultCollection : public algorithms::quality_metric_set::ResultCollection
{
public:
    ResultCollection() {}
    virtual ~ResultCollection() {}

    /**
     * Returns the result of the quality metrics algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    DAAL_EXPORT algorithms::ResultPtr getResult(QualityMetricId id) const;
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 *        specialized for using with the pca algorithm
 */
class InputDataCollection : public algorithms::quality_metric_set::InputDataCollection
{
public:
    InputDataCollection() {}
    virtual ~InputDataCollection() {}

    /**
     * Returns the input object of the quality metrics algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    DAAL_EXPORT algorithms::InputPtr getInput(QualityMetricId id) const;
};
typedef services::SharedPtr<InputDataCollection> InputDataCollectionPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::ResultCollection;
using interface1::InputDataCollection;
using interface1::ResultCollectionPtr;
using interface1::InputDataCollectionPtr;

} // namespace quality_metric_set
/** @} */
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif // __PCA_QUALITY_METRIC_SET_TYPES_H__
