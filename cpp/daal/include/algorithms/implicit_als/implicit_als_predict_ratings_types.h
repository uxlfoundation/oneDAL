/* file: implicit_als_predict_ratings_types.h */
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
//  Implementation of the classes used in the rating prediction stage
//  of the implicit ALS algorithm
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_TYPES_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
/**
 * @defgroup implicit_als_prediction Prediction
 * \copydoc daal::algorithms::implicit_als::prediction
 * @ingroup implicit_als
 * @{
 */
/**
 * \brief Contains classes for making implicit ALS model-based prediction
 */
namespace prediction
{
/**
 * \brief Contains classes for computing ratings based on the implicit ALS model
 */
namespace ratings
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__METHOD"></a>
 * Available methods for computing the results of the implicit ALS model-based prediction
 */
enum Method
{
    defaultDense     = 0, /*!< Default: predicts ratings based on the ALS model and input data in the dense format */
    allUsersAllItems = 0  /*!< Predicts ratings for all users and items based on the ALS model and input data in the dense format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__MODELINPUTID"></a>
 * Available identifiers of input model objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum ModelInputId
{
    model, /*!< %Input model trained by the ALS algorithm */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALMODELINPUTID"></a>
 * Available identifiers of input PartialModel objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum PartialModelInputId
{
    usersPartialModel, /*!< %Input partial model with users factors trained by the implicit ALS algorithm
                                     in the distributed processing mode */
    itemsPartialModel, /*!< %Input partial model with items factors trained by the implicit ALS algorithm
                                     in the distributed processing mode */
    lastPartialModelInputId = itemsPartialModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALRESULTID"></a>
 * Available identifiers of input PartialModel objects for the rating prediction stage
 * of the implicit ALS algorithm
 */
enum PartialResultId
{
    finalResult, /*!< Result of the implicit ALS ratings prediction algorithm */
    lastPartialResultId = finalResult
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RESULTID"></a>
 * Available identifiers of the results of the rating prediction stage of the implicit ALS algorithm
 */
enum ResultId
{
    prediction, /*!< Numeric table with the predicted ratings */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__INPUTIFACE"></a>
 * \brief %Input interface for the rating prediction stage of the implicit ALS algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    InputIface(const InputIface & other) : daal::algorithms::Input(other) {}
    virtual ~InputIface() {}

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    virtual size_t getNumberOfUsers() const = 0;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    virtual size_t getNumberOfItems() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__INPUT"></a>
 * \brief %Input objects for the rating prediction stage of the implicit ALS algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    Input(const Input & other) : InputIface(other) {}
    virtual ~Input() {}

    /**
     * Returns an input Model object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          Input object that corresponds to the given identifier
     */
    ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input Model object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const ModelPtr & ptr);

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    size_t getNumberOfUsers() const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    size_t getNumberOfItems() const DAAL_C11_OVERRIDE;

    /**
     * Checks the input objects and parameters of the implicit ALS algorithm in the rating prediction stage
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the rating prediction stage of the implicit ALS algorithm
 * in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the first step of the rating prediction stage of the implicit ALS algorithm
 * in the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step1Local> : public InputIface
{
public:
    DistributedInput();
    DistributedInput(const DistributedInput & other) : InputIface(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    PartialModelPtr get(PartialModelInputId id) const;

    /**
     * Sets an input object for the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const PartialModelPtr & ptr);

    /**
     * Returns the number of rows in the input numeric table
     * \return Number of rows in the input numeric table
     */
    size_t getNumberOfUsers() const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input numeric table
     * \return Number of columns in the input numeric table
     */
    size_t getNumberOfItems() const DAAL_C11_OVERRIDE;

    /**
     * Checks the parameters of the rating prediction stage of the implicit ALS algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RESULT"></a>
 * \brief Provides methods to access the prediction results obtained with the compute() method
 *        of the implicit ALS algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();
    virtual ~Result() {}

    /**
     * Returns the prediction result of the implicit ALS algorithm
     * \param[in] id   Identifier of the prediction result, \ref ResultId
     * \return         Prediction result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the prediction result of the implicit ALS algorithm
     * \param[in] id    Identifier of the prediction result, \ref ResultId
     * \param[in] ptr   Pointer to the prediction result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store the result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks the result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm in the rating prediction stage */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    /** Default constructor */
    PartialResult();
    /** Default destructor */
    virtual ~PartialResult() {}

    /**
     * Allocates memory to store partial results of the rating prediction stage of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    ResultPtr get(PartialResultId id) const;

    /**
     * Sets a partial result of the rating prediction stage of the implicit ALS algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(PartialResultId id, const ResultPtr & ptr);

    /**
     * Checks a partial result of the implicit ALS algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;

} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::DistributedInput;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace ratings
} // namespace prediction
/** @} */
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
