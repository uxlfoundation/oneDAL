/* file: linear_regression_single_beta_types.h */
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
//  Interface for the linear regression algorithm quality metrics for a single beta coefficient
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "algorithms/linear_regression/linear_regression_model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
/**
 * @defgroup linear_regression_quality_metric_single_beta Single Beta Coefficient
 * \copydoc daal::algorithms::linear_regression::quality_metric_set::single_beta
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
namespace single_beta
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__METHOD"></a>
 * Available methods for computing the quality metrics for a single beta coefficient
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__DATAINPUTID"></a>
* \brief Available identifiers of input objects for a single beta quality metrics
*/
enum DataInputId
{
    expectedResponses,  /*!< NumericTable n x k. Expected responses (Y), dependent variables */
    predictedResponses, /*!< NumericTable n x k. Predicted responses (Z) */
    lastDataInputId = predictedResponses
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__MODELINPUTID"></a>
* \brief Available identifiers of input objects for single beta quality metrics
*/
enum ModelInputId
{
    model            = lastDataInputId + 1, /*!< Linear regression model */
    lastModelInputId = model
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__RESULTID"></a>
* \brief Available identifiers of the result of single beta quality metrics
*/
enum ResultId
{
    rms,                 /*!< NumericTable 1 x k. Root means square errors computed for each response (dependent variable) */
    variance,            /*!< NumericTable 1 x k. Variance computed for each response (dependent variable) */
    zScore,              /*!< NumericTable k x nBeta. Z-score statistics used in testing of insignificance one beta coefficient. H0: beta[i]=0 */
    confidenceIntervals, /*!< NumericTable k x 2 x nBeta. Limits of the confidence intervals for each beta */
    inverseOfXtX,        /*!< NumericTable nBeta x nBeta. Inverse(Xt * X) matrix */
    lastResultId = inverseOfXtX
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__RESULTDATACOLLECTIONID"></a>
* \brief Available identifiers of the result of single beta quality metrics
*/
enum ResultDataCollectionId
{
    betaCovariances =
        lastResultId
        + 1, /*!< DataColection, contains k numeric tables with nBeta x nBeta variance-covariance matrix for betas of each response (dependent variable) */
    lastResultDataCollectionId = betaCovariances
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__PARAMETER"></a>
 * \brief Parameters for the compute() method of single beta quality metrics
 *
 * \snippet linear_regression/linear_regression_single_beta_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(double alphaVal = 0.05, double accuracyVal = 0.001);
    virtual ~Parameter() {}

    double alpha;             /*!< Significance level used in the computation of betas confidence intervals */
    double accuracyThreshold; /*!< Values below this threshold are considered equal to it*/
    /**
    * Checks the correctness of the parameter
    *
    * \return Status of computations
    */
    virtual services::Status check() const;
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__INPUT"></a>
* \brief %Input objects for single beta quality metrics
*/
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    DAAL_CAST_OPERATOR(Input)
    DAAL_DOWN_CAST_OPERATOR(Input, daal::algorithms::Input)

    /** Default constructor */
    Input();

    virtual ~Input() {}

    /**
    * Returns an input object for linear regression quality metric
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DataInputId id) const;

    /**
    * Sets an input object for linear regression quality metric
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(DataInputId id, const data_management::NumericTablePtr & value);

    /**
    * Returns an input object representing linear regression model
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    linear_regression::ModelPtr get(ModelInputId id) const;

    /**
    * Sets an input object representing linear regression model
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ModelInputId id, const linear_regression::ModelPtr & value);

    /**
    * Checks an input object for the linear regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};
typedef services::SharedPtr<Input> InputPtr;

/**
* <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__RESULT"></a>
* \brief Provides interface for the result of linear regression quality metrics
*/
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DAAL_CAST_OPERATOR(Result)
    DAAL_DOWN_CAST_OPERATOR(Result, daal::algorithms::Result)

    Result();

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(ResultDataCollectionId id) const;

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \param[in] index Index in result collection
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultDataCollectionId id, size_t index) const;

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultDataCollectionId id, const data_management::DataCollectionPtr & value);

    /**
    * Allocates memory to store
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Algorithm method
    *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
    * Checks the result of linear regression quality metrics
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE { return SERIALIZATION_LINEAR_REGRESSION_SINGLE_BETA_RESULT_ID; }

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::Result;
using interface1::ResultPtr;
using interface1::Input;
using interface1::InputPtr;

} // namespace single_beta
/** @} */
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif // __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__
