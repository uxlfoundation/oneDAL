/* file: ridge_regression_training_types.h */
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
//  Implementation of the ridge regression algorithm interface
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_TYPES_H__
#define __RIDGE_REGRESSION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/ridge_regression/ridge_regression_model.h"
#include "algorithms/linear_model/linear_model_training_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the ridge regression algorithm
 */
namespace ridge_regression
{
/**
 * @defgroup ridge_regression_training Training
 * \copydoc daal::algorithms::ridge_regression::training
 * @ingroup ridge_regression
 * @{
 */
/**
 * \brief Contains a class for ridge regression model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__RIDGE_REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for ridge regression model-based training
 */
enum Method
{
    defaultDense = 0, /*!< Normal equations method */
    normEqDense  = 0  /*!< Normal equations method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__RIDGE_REGRESSION__TRAINING__INPUTID"></a>
 * \brief Available identifiers of input objects for ridge regression model-based training
 */
enum InputId
{
    data               = linear_model::training::data,               /*!< %Input data table */
    dependentVariables = linear_model::training::dependentVariables, /*!< Values of the dependent variable for the input data */
    lastInputId        = dependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__RIDGE_REGRESSION__TRAINING__MASTER_INPUT_ID"></a>
 * \brief Available identifiers of input objects for ridge regression model-based training in the second step of the distributed processing mode
 */
enum Step2MasterInputId
{
    partialModels, /*!< Collection of partial models trained on local nodes */
    lastStep2MasterInputId = partialModels
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__RIDGE_REGRESSION__TRAINING__PARTIAL_RESULT_ID"></a>
 * \brief Available identifiers of a partial result of ridge regression model-based training
 */
enum PartialResultID
{
    partialModel, /*!< Partial model trained on the available input data */
    lastPartialResultID = partialModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__RIDGE_REGRESSION__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the result of ridge regression model-based training
 */
enum ResultId
{
    model        = linear_model::training::model, /*!< Linear regression model */
    lastResultId = model
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__INPUTIFACE"></a>
 * \brief Abstract class that specifies the interface of input objects for ridge regression model-based training
 */
class InputIface
{
public:
    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    virtual size_t getNumberOfFeatures() const = 0;

    /**
     * Returns the number of dependent variables
     * \return Number of dependent variables
     */
    virtual size_t getNumberOfDependentVariables() const = 0;

    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__INPUT"></a>
 * \brief %Input objects for ridge regression model-based training
 */
class DAAL_EXPORT Input : public linear_model::training::Input, public InputIface
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /** Assignment operator */
    Input & operator=(const Input & other);

    virtual ~Input() {}

    /**
     * Returns an input object for ridge regression model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for ridge regression model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Returns the number of dependent variables
    * \return Number of dependent variables
    */
    size_t getNumberOfDependentVariables() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for the ridge regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input object for ridge regression model-based training in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-RIDGE_REGRESSION__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access a partial result obtained with the compute() method of ridge regression model-based training in the online or
 *        distributed processing mode
 */
class DAAL_EXPORT PartialResult : public linear_model::training::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    PartialResult();

    /**
    * Returns a partial result of ridge regression model-based training
    * \param[in] id    Identifier of the partial result
    * \return          Partial result that corresponds to the given identifier
    */
    daal::algorithms::ridge_regression::ModelPtr get(PartialResultID id) const;

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const;

    /**
    * Returns the number of dependent variables
    * \return Number of dependent variables
    */
    size_t getNumberOfDependentVariables() const;

    /**
     * Sets an argument of the partial result
     * \param[in] id      Identifier of the argument
     * \param[in] value   Pointer to the argument
     */
    void set(PartialResultID id, const daal::algorithms::ridge_regression::ModelPtr & value);

    /**
     * Allocates memory to store a partial result of ridge regression model-based training
     * \param[in] input %Input object for the algorithm
     * \param[in] method Method of ridge regression model-based training
     * \param[in] parameter %Parameter of ridge regression model-based training
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Initializes memory to store a partial result of ridge regression model-based training
     * \param[in] input %Input object for the algorithm
     * \param[in] method Method of ridge regression model-based training
     * \param[in] parameter %Parameter of ridge regression model-based training
     *
     * \return Status of initialization
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the ridge regression algorithm
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks a partial result of the ridge regression algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;
typedef services::SharedPtr<const PartialResult> PartialResultConstPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__DISTRIBUTED_INPUT"></a>
 * \brief %Input object for ridge regression model-based training in the second step of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step2Master> : public daal::algorithms::Input, public InputIface
{
public:
    DistributedInput();
    DistributedInput(const DistributedInput & other);
    DistributedInput operator=(const DistributedInput & other);

    virtual ~DistributedInput() {}

    /**
     * Gets an input object for ridge regression model-based training in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
     * Sets an input object for ridge regression model-based training in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   %Input object
     */
    void set(Step2MasterInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     Adds an input object for ridge regression model-based training in the second step of the distributed processing mode
     * \param[in] id      Identifier of the input object
     * \param[in] partialResult   %Input object
     */
    void add(Step2MasterInputId id, const PartialResultPtr & partialResult);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of dependent variables
     * \return Number of dependent variables
     */
    size_t getNumberOfDependentVariables() const DAAL_C11_OVERRIDE;

    /**
     * Checks an input object for ridge regression model-based training in the second step
     * of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of ridge regression model-based training
 */
class DAAL_EXPORT Result : public linear_model::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of ridge regression model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    ridge_regression::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of ridge regression model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const ridge_regression::ModelPtr & value);

    /**
     * Allocates memory to store the result of ridge regression model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of ridge regression model-based training
     * \param[in] method Computation method for the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method);

    /**
     * Allocates memory to store the result of ridge regression model-based training
     * \param[in] partialResult Pointer to an object containing the input data
     * \param[in] method        Computation method of the algorithm
     * \param[in] parameter     %Parameter of ridge regression model-based training
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, const Parameter * parameter, int method);

    /**
     * Checks the result of ridge regression model-based training
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks the result of the ridge regression model-based training
     * \param[in] pr      %PartialResult of the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;
} // namespace interface1

using interface1::InputIface;
using interface1::Input;
using interface1::DistributedInput;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::PartialResultConstPtr;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace training
} // namespace ridge_regression
/** @} */
} // namespace algorithms
} // namespace daal

#endif
