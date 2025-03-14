/* file: linear_regression_group_of_betas_types.h */
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
//  Interface for the linear regression algorithm quality metrics for a group of beta coefficients
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_TYPES_H__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
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
 * @defgroup linear_regression_quality_metric_group_of_betas Group of Beta Coefficients
 * \copydoc daal::algorithms::linear_regression::quality_metric_set::group_of_betas
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
namespace group_of_betas
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__METHOD"></a>
 * Available methods for computing the quality metrics for a group of beta coefficients
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__DATAINPUTID"></a>
* \brief Available identifiers of input objects for a group of betas quality metrics
*/
enum DataInputId
{
    expectedResponses,              /*!< NumericTable n x k. Expected responses (Y), dependent variables */
    predictedResponses,             /*!< NumericTable n x k. Predicted responses (Z)  */
    predictedReducedModelResponses, /*!< NumericTable n x k. Responses predicted by reduced model where p - p0 of p betas are set to zero */
    lastDataInputId = predictedReducedModelResponses
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__RESULTID"></a>
* \brief Available identifiers of the result of a group of betas quality metrics
*/
enum ResultId
{
    expectedMeans,      /*!< NumericTable 1 x k. Means of expected responses computed for each dependent variable */
    expectedVariance,   /*!< NumericTable 1 x k. Variance of expected responses computed for each dependent variable */
    regSS,              /*!< NumericTable 1 x k. Regression sum of squares computed for each dependent variable */
    resSS,              /*!< NumericTable 1 x k. Sum of squares of residuals computed for each dependent variable */
    tSS,                /*!< NumericTable 1 x k. Total sum of squares of residuals computed for each dependent variable */
    determinationCoeff, /*!< NumericTable 1 x k. Determination coefficient computed for each dependent variable */
    fStatistics,        /*!< NumericTable 1 x k. F-statistics computed for each dependent variable */
    lastResultId = fStatistics
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__PARAMETER"></a>
 * \brief Parameters for the compute() method of a group of betas quality metrics
 *
 * \snippet linear_regression/linear_regression_group_of_betas_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nBeta, size_t nBetaReducedModel) : numBeta(nBeta), numBetaReducedModel(nBetaReducedModel), accuracyThreshold(0.001) {}
    virtual ~Parameter() {}

    size_t numBeta; /*!< Number of beta coefficients (p) of linear regression model used for prediction */
    size_t
        numBetaReducedModel; /*!< Number of beta coefficients (p0) used for prediction with reduced linear regression model where p - p0 of p beta coefficients are set to 0 */
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
* <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__INPUT"></a>
* \brief %Input objects for a group of betas quality metrics
*/
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    DAAL_CAST_OPERATOR(Input)
    DAAL_DOWN_CAST_OPERATOR(Input, daal::algorithms::Input)

    /** Default constructor */
    Input() : daal::algorithms::Input(lastDataInputId + 1) {}

    virtual ~Input() {}

    /**
    * Returns an input object for linear regression quality metric
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DataInputId id) const { return data_management::NumericTable::cast(Argument::get(id)); }

    /**
    * Sets an input object for linear regression quality metric
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(DataInputId id, const data_management::NumericTablePtr & value) { Argument::set(id, value); }

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
* <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUP_OF_BETAS__RESULT"></a>
* \brief Provides interface for the result of linear regression quality metrics
*/
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DAAL_CAST_OPERATOR(Result)
    DAAL_DOWN_CAST_OPERATOR(Result, daal::algorithms::Result)

    Result() : daal::algorithms::Result(lastResultId + 1) {};

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const { return data_management::NumericTable::cast(Argument::get(id)); }

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultId id, const data_management::NumericTablePtr & value) { Argument::set(id, value); }

    /**
    * Allocates memory to store
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Algorithm method
    *
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
    {
        const data_management::NumericTable * dependentVariableTable = (static_cast<const Input *>(input))->get(expectedResponses).get();
        const size_t nDepVariable                                    = dependentVariableTable->getNumberOfColumns();
        services::Status status;
        for (size_t i = 0; i < (lastResultId + 1); ++i)
        {
            Argument::set(i, data_management::HomogenNumericTable<algorithmFPType>::create(
                                 nDepVariable, 1, data_management::NumericTableIface::doAllocate, 0, &status));
        }
        return status;
    }

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
    int getSerializationTag() const DAAL_C11_OVERRIDE { return SERIALIZATION_LINEAR_REGRESSION_GROUP_OF_BETAS_RESULT_ID; }

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
using interface1::Input;
using interface1::ResultPtr;
using interface1::InputPtr;

} // namespace group_of_betas
/** @} */
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif // __LINEAR_REGRESSION_GROUP_OF_BETAS_TYPES_H__
