/* file: svm_predict_types.h */
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
//  SVM parameter structure
//--
*/

#ifndef __SVM_PREDICT_TYPES_H__
#define __SVM_PREDICT_TYPES_H__

#include "algorithms/classifier/classifier_predict_types.h"
#include "algorithms/svm/svm_model.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
/**
 * @defgroup svm_prediction Prediction
 * \copydoc daal::algorithms::svm::prediction
 * @ingroup svm
 * @{
 */
/**
 * \brief Contains classes to make predictions based on the SVM model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVM__PREDICTION__METHOD"></a>
 * Available methods to run predictions based on the SVM model
 */
enum Method
{
    defaultDense = 0 /*!< Default SVM model-based prediction method */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the svm algorithm
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
     * Returns the input Model object in the prediction stage of the SVM algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    svm::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the SVM algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const svm::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computation
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1

using interface1::Input;

} // namespace prediction
/** @} */
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
