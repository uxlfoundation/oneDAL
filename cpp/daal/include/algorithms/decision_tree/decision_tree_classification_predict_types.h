/* file: decision_tree_classification_predict_types.h */
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
//  Implementation of the Decision tree classification algorithm interface
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_PREDICT_TYPES_H__
#define __DECISION_TREE_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the Decision tree algorithm
 */
namespace decision_tree
{
/**
 * \brief Contains classes of the Decision tree classification algorithm
 */
namespace classification
{
/**
 * @defgroup decision_tree_classification_prediction Prediction
 * \copydoc daal::algorithms::decision_tree::classification::prediction
 * @ingroup decision_tree_classification
 * @{
 */
/**
 * \brief Contains a class for making Decision tree model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__METHOD"></a>
 * \brief Available methods for making Decision tree model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making Decision tree model-based prediction
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;

public:
    /** Default constructor */
    Input();
    Input(const Input & other) : super(other) {}

    using super::get;
    using super::set;

    /**
     * Returns the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(classifier::prediction::ModelInputId id, const ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

typedef services::SharedPtr<Input> InputPtr;
typedef services::SharedPtr<const Input> InputConstPtr;

} // namespace interface1

using interface1::Input;
using interface1::InputPtr;
using interface1::InputConstPtr;

} // namespace prediction
/** @} */
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
