/* file: multinomial_naive_bayes_predict_types.h */
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
//  Naive Bayes classifier parameter structure used in the prediction stage
//--
*/

#ifndef __NAIVE_BAYES_PREDICT_TYPES_H__
#define __NAIVE_BAYES_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "data_management/data/data_collection.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
/**
 * @defgroup multinomial_naive_bayes_prediction Prediction
 * \copydoc daal::algorithms::multinomial_naive_bayes::prediction
 * @ingroup multinomial_naive_bayes
 * @{
 */
/**
 * \brief Contains classes for multinomial naive Bayes model based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__METHOD"></a>
 * Available methods for computing the results of the naive Bayes model based prediction
 */
enum Method
{
    defaultDense = 0, /*!< Default multinomial naive Bayes model based prediction  */
    fastCSR      = 1  /*!< Multinomial naive Bayes model based prediction for sparse data in CSR format */
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the multinomial naive Bayes algorithm
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    DAAL_DEPRECATED Input();
    Input(const Input & other);
    virtual ~Input() {}

    using super::get;
    using super::set;

    /**
     * Returns the input Numeric Table object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the multinomial naive Bayes algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    multinomial_naive_bayes::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the multinomial naive Bayes algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const multinomial_naive_bayes::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Input;

} // namespace prediction
/** @} */
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
