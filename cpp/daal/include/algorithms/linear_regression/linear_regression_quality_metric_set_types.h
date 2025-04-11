/* file: linear_regression_quality_metric_set_types.h */
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
//  Interface for the linear regression algorithm quality metrics
//--
*/

#ifndef __LINEAR_REGRESSION_QUALITY_METRIC_SET_TYPES_H__
#define __LINEAR_REGRESSION_QUALITY_METRIC_SET_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "algorithms/algorithm_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
/**
 * @defgroup linear_regression_quality_metric_set Quality Metrics
 * \copydoc daal::algorithms::linear_regression::quality_metric_set
 * @ingroup linear_regression
 * @{
 */
namespace quality_metric_set
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__QUALITYMETRICID"></a>
 * Available identifiers of the quality metrics available for the model trained with linear regression algorithm
 */
enum QualityMetricId
{
    singleBeta,   /*!< Single coefficient metrics*/
    groupOfBetas, /*!< Group of coefficients metrics*/
    lastQualityMetricId = groupOfBetas
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
* <a name="DAAL-STRUCT-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__PARAMETER"></a>
* \brief Parameters for the quality metrics set compute() method
*
* \snippet linear_regression/linear_regression_quality_metric_set_types.h Parameter source code
*/
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nBeta, size_t nBetaReducedModel, double alphaVal = 0.05, double accuracyVal = 0.001);

    virtual ~Parameter() {}

    double alpha;             /*!< Significance level used in the computation of betas confidence intervals */
    double accuracyThreshold; /*!< Values below this threshold are considered equal to it*/
    size_t numBeta;           /*!< Number of beta coefficients (p) of linear regression model used for prediction */
    size_t
        numBetaReducedModel; /*!< Number of beta coefficients (p0) used for prediction with reduced linear regression model where p - p0 of p beta coefficients are set to 0 */

    /**
    * Checks the correctness of the parameter
    *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 *        specialized for using with the linear regression training algorithm
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
    algorithms::ResultPtr getResult(QualityMetricId id) const;
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 *        specialized for using with the linear regression training algorithm
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
    algorithms::InputPtr getInput(QualityMetricId id) const;
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
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif // __LINEAR_REGRESSION_QUALITY_METRIC_SET_TYPES_H__
