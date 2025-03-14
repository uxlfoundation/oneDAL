/* file: stump_classification_predict_types.h */
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

#ifndef __STUMP_CLASSIFICATION_PREDICT_TYPES_H__
#define __STUMP_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/stump/stump_classification_model.h"

#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the Decision stump algorithm
 */
namespace stump
{
/**
 * \brief Contains classes of the Decision stump classification algorithm
 */
namespace classification
{
/**
 * @defgroup stump_classification_prediction Prediction
 * \copydoc daal::algorithms::stump::classification::prediction
 * @ingroup stump_classification
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the stump algorithm
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    Input();
    Input(const Input & other);

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
     * Returns the input Model object in the prediction stage of the Stump algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    stump::classification::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Stump algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const stump::classification::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Input;
} // namespace prediction
/** @} */
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif // __stump__classification_PREDICT_TYPES_H__
