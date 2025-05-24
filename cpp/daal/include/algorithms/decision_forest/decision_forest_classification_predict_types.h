/* file: decision_forest_classification_predict_types.h */
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

#ifndef __DECISION_FOREST_CLASSIFICATION_PREDICT_TYPES_H__
#define __DECISION_FOREST_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
/**
 * @defgroup decision_forest_classification_prediction Prediction
 * \copydoc daal::algorithms::decision_forest::classification::prediction
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the classifier model */
namespace prediction
{
/**
* <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__PREDICTION__VOTING_METHOD"></a>
* \brief Available methods for averaging trees predictions
*/
enum VotingMethod
{
    weighted = 0,
    unweighted,
    lastResultId = unweighted
};
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the DECISION_FOREST_CLASSIFICATION algorithm
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
     * Returns the input Model object in the prediction stage of the DECISION_FOREST_CLASSIFICATION algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    decision_forest::classification::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the DECISION_FOREST_CLASSIFICATION algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const decision_forest::classification::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_FOREST_CLASSIFICATION__PREDICTION__PARAMETER"></a>
 * \brief Class for the parameters of the Decision Forest classification algorithm
 *
 * \snippet decision_forest/decision_forest_classification_predict_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    Parameter(size_t nClasses, VotingMethod votingMethod = weighted) : daal::algorithms::classifier::Parameter(nClasses), votingMethod(votingMethod)
    {}
    VotingMethod votingMethod;
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Input;
using interface1::Parameter;
} // namespace prediction
/** @} */
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif // __DECISION_FOREST_CLASSIFICATION_PREDICT_TYPES_H__
