/* file: multi_class_classifier_predict_types.h */
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
//  Multiclass classifier data types
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_PREDICT_TYPES_H__
#define __MULTI_CLASS_CLASSIFIER_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * @defgroup multi_class_classifier_prediction Prediction
 * \copydoc daal::algorithms::multi_class_classifier::prediction
 * @ingroup multi_class_classifier
 * @{
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__METHOD"></a>
 * Computation methods for the multi-class classifier prediction algorithm
 */
enum Method
{
    defaultDense           = 0, /*!< Default: Prediction method for the multi-class classifier proposed by Ting-Fan Wu et al. */
    multiClassClassifierWu = 0, /*!< Prediction method for the multi-class classifier proposed by Ting-Fan Wu et al. */
    voteBased              = 1  /*!< Prediction method that is based on votes returned by two-class classifiers */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for the multi-class classifier prediction
 */
enum ResultId
{
    prediction       = classifier::prediction::prediction,       /*!< Prediction results */
    decisionFunction = classifier::prediction::lastResultId + 1, /*!< Decision function */
    lastResultId     = decisionFunction
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the Multi-class classifier algorithm
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
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(classifier::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the Multi-class classifier algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    multi_class_classifier::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Multi-class classifier algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::ModelInputId id, const multi_class_classifier::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computation
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const override;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__RESULT"></a>
* \brief Provides interface for the result of model-based prediction
*/
class DAAL_EXPORT Result : public classifier::prediction::Result
{
    typedef classifier::prediction::Result super;

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
    * \param[in] input    %Input object
    * \param[in] par      %Parameter of the algorithm
    * \param[in] pmethod  Computation method for the algorithm, \ref prediction::Method
    * \param[in] tmethod  Computation method that was used to train the multi-class classifier model, \ref training::Method
    * \return Status of allocation
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int pmethod,
                                          const int tmethod);

    /**
    * Checks the result of model-based prediction
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const override;

protected:
    using classifier::prediction::Result::check;

    /** \protected */
    services::Status checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const;

    /** \protected */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::classifier::prediction::Result::serialImpl<Archive, onDeserialize>(arch);
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
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
