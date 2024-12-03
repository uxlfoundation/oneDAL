/* file: logitboost_predict_types.h */
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

#ifndef __LOGITBOOST_PREDICT_TYPES_H__
#define __LOGITBOOST_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/boosting/logitboost_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
/**
 * @defgroup logitboost_prediction Prediction
 * \copydoc daal::algorithms::logitboost::prediction
 * @ingroup logitboost
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{
/**
 * \brief Contains version 2.0 of the Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the logitboost algorithm
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    Input() : classifier::prediction::Input() {}
    Input(const Input & other) : classifier::prediction::Input(other) {}
    Input & operator=(const Input & other) = default;
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
     * Returns the input Model object in the prediction stage of the LogitBoost algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    logitboost::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the LogitBoost algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const logitboost::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface2
using interface2::Input;
} // namespace prediction
/** @} */
} // namespace logitboost
} // namespace algorithms
} // namespace daal
#endif // __LOGITBOOST_PREDICT_TYPES_H__
