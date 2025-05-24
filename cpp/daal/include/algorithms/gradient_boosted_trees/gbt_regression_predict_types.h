/* file: gbt_regression_predict_types.h */
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
//  Implementation of the gradient boosted trees algorithm interface
//--
*/

#ifndef __GBT_REGRESSSION_PREDICT_TYPES_H__
#define __GBT_REGRESSSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/regression/regression_predict_types.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the gradient boosted trees algorithm
 */
namespace gbt
{
namespace regression
{
/**
 * @defgroup gbt_regression_prediction Prediction
 * \copydoc daal::algorithms::gbt::regression::prediction
 * @ingroup gbt_regression
 * @{
 */
/**
 * \brief Contains a class for making model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSSION__PREDICTION__METHOD"></a>
 * \brief Available methods for making model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSSION__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making model-based prediction
 */
enum NumericTableInputId
{
    data                    = algorithms::regression::prediction::data, /*!< Input data table */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__PREDICTION__REGRESSSION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making model-based prediction
 */
enum ModelInputId
{
    model            = algorithms::regression::prediction::model, /*!< Trained gradient boosted trees model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__PREDICTION__REGRESSSION__RESULTID"></a>
 * \brief Available identifiers of the result for making model-based prediction
 */
enum ResultId
{
    prediction   = algorithms::regression::prediction::prediction, /*!< Result of gradient boosted trees model-based prediction */
    lastResultId = prediction
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__PREDICTION__REGRESSSION__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute - results are mutually exclusive
 */
enum ResultToComputeId
{
    predictionResult  = (1 << 0), /*!< Compute the regular prediction */
    shapContributions = (1 << 1), /*!< Compute SHAP contribution values */
    shapInteractions  = (1 << 2)  /*!< Compute SHAP interaction values */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__GBT__REGRESSION__PREDICTION__PARAMETER"></a>
 * \brief Parameters of the prediction algorithm
 *
 * \snippet gradient_boosted_trees/gbt_regression_predict_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    typedef daal::algorithms::Parameter super;

    Parameter() : super(), nIterations(0), resultsToCompute(predictionResult) {}
    Parameter(const Parameter & o)             = default;
    Parameter & operator=(const Parameter & o) = default;
    size_t nIterations;           /*!< Number of iterations of the trained model to be uses for prediction*/
    DAAL_UINT64 resultsToCompute; /*!< 64 bit integer flag that indicates the results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making model-based prediction
 */
class DAAL_EXPORT Input : public algorithms::regression::prediction::Input
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);
    virtual ~Input() {}

    /**
     * Returns an input object for making model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns an input object for making model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    gbt::regression::ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input object for making model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & value);

    /**
     * Sets an input object for making model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ModelInputId id, const gbt::regression::ModelPtr & value);

    /**
     * Checks an input object for making model-based prediction
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of model-based prediction
 */
class DAAL_EXPORT Result : public algorithms::regression::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store a partial result of model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace prediction
/** @} */
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
