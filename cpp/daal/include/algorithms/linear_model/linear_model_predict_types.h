/* file: linear_model_predict_types.h */
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
//  Implementation of the regression algorithm interface
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_TYPES_H__
#define __LINEAR_MODEL_PREDICT_TYPES_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_types.h"
#include "algorithms/linear_model/linear_model_model.h"
#include "algorithms/regression/regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
/**
 * @defgroup linear_model_prediction Prediction
 * \copydoc daal::algorithms::linear_model::prediction
 * @ingroup linear_model
 * @{
 */
/**
 * \brief Contains a class for making the regression model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__PREDICTION__METHOD"></a>
 * \brief Available methods for making the regression model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making the regression model-based prediction
 */
enum NumericTableInputId
{
    data                    = regression::prediction::data, /*!< Input data table */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making the regression model-based prediction
 */
enum ModelInputId
{
    model            = regression::prediction::model, /*!< Trained regression model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making the regression model-based prediction
 */
enum ResultId
{
    prediction   = regression::prediction::prediction, /*!< Result of the regression model-based prediction */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making the regression model-based prediction
 */
class DAAL_EXPORT Input : public regression::prediction::Input
{
public:
    /** Default constructor */
    Input(size_t nElements = 0);
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input() {}

    /**
     * Returns an input object for making the regression model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns an input object for making the regression model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    linear_model::ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input object for making the regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & value);

    /**
     * Sets an input object for making the regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ModelInputId id, const linear_model::ModelPtr & value);

    /**
     * Checks an input object for making the regression model-based prediction
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of the regression model-based prediction
 */
class DAAL_EXPORT Result : public regression::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result(size_t nElements = 0);

    /**
     * Returns the result of the regression model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the regression model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store a partial result of the regression model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of the regression model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return regression::prediction::Result::serialImpl<Archive, onDeserialize>(arch);
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
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
