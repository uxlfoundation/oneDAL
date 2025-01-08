/* file: gbt_classification_predict_types.h */
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
//  Implementation of the base classes used in the prediction stage
//  of the classifier algorithm
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_TYPES_H__
#define __GBT_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
/**
 * @defgroup gbt_classification_prediction Prediction
 * \copydoc daal::algorithms::gbt::classification::prediction
 * @ingroup gbt_classification
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__METHOD"></a>
 * Available methods for predictions based on the gbt model
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making model-based prediction
 */
enum ModelInputId
{
    model            = algorithms::classifier::prediction::model, /*!< Trained gradient boosted trees model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making model-based prediction
 */
enum ResultId
{
    prediction       = algorithms::classifier::prediction::prediction,
    probabilities    = algorithms::classifier::prediction::probabilities,
    logProbabilities = algorithms::classifier::prediction::logProbabilities,
    lastResultId     = logProbabilities
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute - results are mutually exclusive
 */
enum ResultToComputeId
{
    predictionResult  = (1 << 0), /*!< Compute the regular prediction */
    shapContributions = (1 << 1), /*!< Compute SHAP contribution values */
    shapInteractions  = (1 << 2)  /*!< Compute SHAP interaction values */
};

/**
 * \brief Contains version 2.0 of the Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__PARAMETER"></a>
 * \brief Parameters of the prediction algorithm
 *
 * \snippet gradient_boosted_trees/gbt_classification_predict_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    typedef daal::algorithms::classifier::Parameter super;

    Parameter(size_t nClasses = 2) : super(nClasses), nIterations(0), resultsToCompute(predictionResult) {}
    Parameter(const Parameter & o)             = default;
    Parameter & operator=(const Parameter & o) = default;
    size_t nIterations;           /*!< Number of iterations of the trained model to be used for prediction */
    DAAL_UINT64 resultsToCompute; /*!< 64 bit integer flag that indicates the results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__RESULT"></a>
 * \brief Provides interface for the result of model-based prediction
 */
class DAAL_EXPORT Result : public algorithms::classifier::prediction::Result
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
    using daal::algorithms::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface2

/**
 * \brief Contains version 1.0 of the Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the GBT_CLASSIFICATION algorithm
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);
    virtual ~Input() {}

    using super::get;
    using super::set;

    /**
     * Returns the input Numeric Table object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the GBT_CLASSIFICATION algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    gbt::classification::ModelPtr get(ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the GBT_CLASSIFICATION algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const gbt::classification::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface2::Parameter;
using interface2::Result;
using interface2::ResultPtr;
using interface1::Input;
} // namespace prediction
/** @} */
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif // __GBT_CLASSIFICATION_PREDICT_TYPES_H__
