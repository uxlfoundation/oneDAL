/* file: logistic_regression_predict_types.h */
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

#ifndef __LOGISTIC_REGRESSION_PREDICT_TYPES_H__
#define __LOGISTIC_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
/**
 * @defgroup logistic_regression_prediction Prediction
 * \copydoc daal::algorithms::logistic_regression::prediction
 * @ingroup logistic_regression
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__METHOD"></a>
 * Available methods for predictions based on the logistic regression model
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__RESULTTOCOMPUTEID"></a>
* Available identifiers to specify the result to compute
*/
enum ResultToComputeId
{
    computeClassesLabels           = classifier::computeClassLabels,
    computeClassesProbabilities    = classifier::computeClassProbabilities,
    computeClassesLogProbabilities = classifier::computeClassLogProbabilities
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__RESULT_NUMERIC_TABLE_ID"></a>
* Available identifiers of results obtained in the prediction stage of the classification algorithm
*/
enum ResultNumericTableId
{
    probabilities    = classifier::prediction::probabilities,    /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                     containing probabilities of classes computed when
                                                                     computeClassesProbabilities option is enabled.
                                                                     In case  nClasses = 2 the table contains probabilities of class _1. */
    logProbabilities = classifier::prediction::logProbabilities, /*!< Numeric table of size: n x 1, if nClasses = 2, n x nClasses, if nClasses > 2
                                                                     containing logarithms of classes_ probabilities computed when
                                                                     computeClassesLogProbabilities option is enabled.
                                                                     In case nClasses = 2 the table contains logarithms of class _1_ probabilities. */
    lastResultNumericTableId = logProbabilities
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the LOGISTIC_REGRESSION algorithm
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    Input() : super() {}
    Input(const Input & other) : super(other) {}
    virtual ~Input() {}

    using super::get;
    using super::set;

    /**
     * Returns the input Numeric Table object in the prediction stage of the algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the LOGISTIC_REGRESSION algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    logistic_regression::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the LOGISTIC_REGRESSION algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const logistic_regression::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Input;
using classifier::Parameter;             /* Support of static backward compatibility */
using classifier::prediction::Result;    /* Support of static backward compatibility */
using classifier::prediction::ResultPtr; /* Support of static backward compatibility */
} // namespace prediction
/** @} */
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif // __LOGISTIC_REGRESSION_PREDICT_TYPES_H__
