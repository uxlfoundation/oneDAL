/* file: classifier_training_types.h */
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
//  Implementation of the base classes used in the training stage
//  of the classification algorithms
//--
*/

#ifndef __CLASSIFIER_TRAINING_TYPES_H__
#define __CLASSIFIER_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup classifier Base Classifier
 * \brief Contains base classes for working with classifiers
 * @ingroup classification
 */
/**
 * \brief Contains classes for working with classifiers
 */
namespace classifier
{
/**
 * @defgroup training Training
 * \copydoc daal::algorithms::classifier::training
 * @ingroup classifier
 * @{
 */
/**
 * \brief Contains classes for training the model of the classification algorithms
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__INPUTID"></a>
 * Available identifiers of the results in the training stage of the classification algorithms
 */
enum InputId
{
    data,    /*!< Training data set */
    labels,  /*!< Labels of the training data set */
    weights, /*!< Optional. Weights of the observations in the training data set */
    lastInputId = weights
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__PARTIALRESULTID"></a>
 * Available identifiers of partial results
 */
enum PartialResultId
{
    partialModel, /*!< Trained partial model */
    lastPartialResultId = partialModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CLASSIFIER__TRAINING__RESULTID"></a>
 * Available identifiers of the results in the training stage of the classification algorithms
 */
enum ResultId
{
    model, /*!< Resulting model */
    lastResultId = model
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__INPUTIFACE"></a>
 * \brief Abstract class that specifies the interface of the classes of the classification algorithm input objects
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements);
    InputIface(const InputIface & other);
    InputIface & operator=(const InputIface & other);
    virtual ~InputIface() {}
    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__INPUT"></a>
 * \brief Base class for the input objects in the training stage of the classification algorithms
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input(size_t nElements = lastInputId + 1);
    Input(const Input & other);
    Input & operator=(const Input & other);
    virtual ~Input() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Returns the input object in the training stage of the classification algorithm
     * \param[in] id   Identifier of the input object, \ref InputId
     * \return         Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object in the training stage of the classification algorithm
     * \param[in] id    Identifier of the input object, \ref InputId
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the classifier training algorithm
 *        in the online or distributed processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    PartialResult();
    virtual ~PartialResult() {}

    /**
     * Returns the partial result in the training stage of the classification algorithm
     * \param[in] id   Identifier of the partial result, \ref PartialResultId
     * \return         Partial result that corresponds to the given identifier
     */
    classifier::ModelPtr get(PartialResultId id) const;

    /**
     * Sets the partial result in the training stage of the classification algorithm
     * \param[in] id    Identifier of the partial result, \ref PartialResultId
     * \param[in] value Pointer to the partial result
     */
    void set(PartialResultId id, const daal::algorithms::classifier::ModelPtr & value);

    /**
     * Checks the correctness of the PartialResult object
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    services::Status checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const;
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method in the
 *        batch processing mode or finalizeCompute() method
 *        in the online or distributed processing mode of the classification algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \return          Model trained with the classification algorithm
     */
    daal::algorithms::classifier::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of the training stage of the classification algorithm
     * \param[in] id    Identifier of the result, \ref ResultId
     * \param[in] value Pointer to the training result
     */
    void set(ResultId id, const daal::algorithms::classifier::ModelPtr & value);

    /**
     * Checks the correctness of the Result object
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::Result::check;

    Result(const size_t n);
    services::Status checkImpl(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
