/* file: implicit_als_training_init_types.h */
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
//  Implementation of the implicit ALS algorithm interface
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_INIT_TYPES_H__
#define __IMPLICIT_ALS_TRAINING_INIT_TYPES_H__

#include "algorithms/implicit_als/implicit_als_model.h"
#include "algorithms/implicit_als/implicit_als_training_types.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
/**
 * @defgroup implicit_als_init Initialization
 * \copydoc daal::algorithms::implicit_als::training::init
 * @ingroup implicit_als_training
 * @{
 */
/**
 * \brief Contains classes for the implicit ALS initialization algorithm
 */
namespace init
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__METHOD"></a>
 * \brief Available methods for initializing the implicit ALS algorithm
 */
enum Method
{
    defaultDense = 0, /*!< Default: initialization method for input data stored in the dense format */
    fastCSR      = 1  /*!< Initialization method for input data stored in the compressed sparse row (CSR) format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INPUTID"></a>
 * \brief Available identifiers of input objects for the implicit ALS initialization algorithm
 */
enum InputId
{
    data, /*!< %Input data table that contains ratings */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__STEP2LOCALINPUTID"></a>
 * Available identifiers of input objects for the implicit ALS initialization algorithm in the second step
 * of the distributed processing mode
 */
enum Step2LocalInputId
{
    inputOfStep2FromStep1, /*!< Partial results of the implicit ALS initialization algorithm computed in the first step
                                     and to be transferred to the second step of the distributed initialization algorithm */
    lastStep2LocalInputId = inputOfStep2FromStep1
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULTBASEID"></a>
 * \brief Available identifiers of partial results of the implicit ALS initialization algorithm in the first and second steps
 * of the distributed processing mode
 */
enum PartialResultBaseId
{
    outputOfInitForComputeStep3, /*!< Partial results of the implicit ALS initialization algorithm computed in the first step
                                             and to be transferred to the third step of the distributed ALS training algorithm */
    offsets,                     /*!< Collection of 1x1 numeric tables that hold the global indices of the starting rows
                                             of the partial models */
    lastPartialResultBaseId = offsets
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the implicit ALS initialization algorithm in the first step
 * of the distributed processing mode
 */
enum PartialResultId
{
    partialModel        = lastPartialResultBaseId + 1, /*!< Partial implicit ALS model */
    lastPartialResultId = partialModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULTCOLLECTIONID"></a>
 * \brief Available identifiers of partial results of the implicit ALS initialization algorithm in the first step
 * of the distributed processing mode
 */
enum PartialResultCollectionId
{
    outputOfStep1ForStep2 = lastPartialResultId + 1, /*!< Partial results of the implicit ALS initialization algorithm computed in the first step
                                     and to be transferred to the second step of the distributed initialization algorithm */
    lastPartialResultCollectionId = outputOfStep1ForStep2
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Available types of partial results of the implicit ALS initialization algorithm in the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    transposedData                      = lastPartialResultBaseId + 1, /*!< CSR numeric table that holds a block of users from the input data set */
    lastDistributedPartialResultStep2Id = transposedData
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__RESULTID"></a>
 * \brief Available identifiers of the results of the implicit ALS initialization algorithm
 */
enum ResultId
{
    model, /*!< Implicit ALS model */
    lastResultId = model
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARAMETER"></a>
 * \brief Parameters of the compute() method of the implicit ALS initialization algorithm
 *
 * \snippet implicit_als/implicit_als_training_init_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs parameters of the implicit ALS initialization algorithm
     * \param[in] nFactors      Total number of factors
     * \param[in] fullNUsers    Full number of users
     * \param[in] seed          Seed for random numbers generation \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(size_t nFactors = 10, size_t fullNUsers = 0, size_t seed = 777777);

    size_t nFactors;           /*!< Total number of factors */
    size_t fullNUsers;         /*!< Full number of users */
    size_t seed;               /*!< Seed for generating random numbers in the initialization step \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine; /*!< Engine for generating random numbers in the initialization step */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDPARAMETER"></a>
 * \brief Parameters of the compute() method of the implicit ALS initialization algorithm in the distributed computing mode
 *
 * \snippet implicit_als/implicit_als_training_init_types.h Distributed parameter source code
 */
/* [Distributed parameter source code] */
struct DAAL_EXPORT DistributedParameter : public Parameter
{
    /**
     * Constructs parameters of the implicit ALS initialization algorithm in the distributed computing mode
     * \param[in] nFactors      Total number of factors
     * \param[in] fullNUsers    Full number of users
     * \param[in] seed          Seed for random numbers generation \DAAL_DEPRECATED_USE{ engine }
     */
    DistributedParameter(size_t nFactors = 10, size_t fullNUsers = 0, size_t seed = 777777);

    data_management::NumericTablePtr partition; /*!< Numeric table od size 1x1 that provides
            the number of input data parts used in implicit ALS initialization algorithm in the distributed computing mode.
            Or, numeric table of size (nParts + 1)x1, where nParts is the number of input data parts used
            in implicit ALS initialization algorithm in the distributed computing mode;
            and partition[i] holds the offset i-th data part to be computed by the initialization algorithm */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Distributed parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INPUT"></a>
 * \brief %Input objects for the implicit ALS initialization algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input(size_t nElements = lastInputId + 1);
    Input(const Input & other) : daal::algorithms::Input(other) {}

    virtual ~Input() {}

    /**
     * Returns an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the number of items, that is, the number of columns in the input data set
     * \return Number of items
     */
    size_t getNumberOfItems() const;

    /**
     * Checks the input objects and parameters of the implicit ALS initialization algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS initialization algorithm in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the implicit ALS initialization algorithm in the first step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step1Local> : public Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : Input(other) {}

    virtual ~DistributedInput() {}

    using Input::get;
    using Input::set;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the implicit ALS initialization algorithm in the second step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step2Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(Step2LocalInputId id) const;

    /**
     * Sets an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(Step2LocalInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Checks the input objects and parameters of the implicit ALS initialization algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULTBASE"></a>
 * \brief Provides interface to access partial results obtained with the implicit ALS initialization algorithm
 * in the first and second steps of the distributed processing mode
 */
class DAAL_EXPORT PartialResultBase : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResultBase)

    /**
     * Constructs partial result of the implicit ALS initialization algorithm
     * with the specified number of elements
     * \param[in] nElements Number of elements
     */
    PartialResultBase(size_t nElements = 0);

    /**
     * Gets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \return Partial result that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(PartialResultBaseId id) const;

    /**
     * Gets a numeric table object from a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] key   Key to use to retrieve a numeric table
     * \return Numeric table from the partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultBaseId id, size_t key) const;

    /**
     * Sets the partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   The partial result object
     */
    void set(PartialResultBaseId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Checks a partial result of the implicit ALS initialization algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::PartialResult::check;

    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(size_t nParts);

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    services::Status checkImpl(size_t nParts) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
class DAAL_EXPORT PartialResult : public PartialResultBase
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    /** Default constructor */
    PartialResult();

    using PartialResultBase::get;
    using PartialResultBase::set;

    /**
     * Allocates memory for storing partial results of the implicit ALS initialization algorithm
     * \param[in] input         Pointer to the input object structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Gets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    PartialModelPtr get(PartialResultId id) const;

    /**
     * Sets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(PartialResultId id, const PartialModelPtr & ptr);

    /**
     * Gets a KeyValueDataCollection partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \return Partial result that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(PartialResultCollectionId id) const;

    /**
     * Gets a numeric table object from a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] key   Key to use to retrieve a numeric table
     * \return Numeric table from the partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultCollectionId id, size_t key) const;

    /**
     * Sets the KeyValueDataCollection partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   The partial result object
     */
    void set(PartialResultCollectionId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Checks a partial result of the implicit ALS initialization algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using PartialResultBase::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public PartialResultBase
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep2)
    /** Default constructor */
    DistributedPartialResultStep2();

    using PartialResultBase::get;
    using PartialResultBase::set;

    /**
     * Allocates memory for storing partial results of the implicit ALS initialization algorithm
     * \param[in] input         Pointer to the input object structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Gets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Sets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   The partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks a partial result of the implicit ALS initialization algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using PartialResultBase::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__RESULT"></a>
 * \brief Provides methods to access the results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::implicit_als::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)

    /**
     * Returns the result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    daal::algorithms::implicit_als::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const daal::algorithms::implicit_als::ModelPtr & ptr);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<PartialResult> PartialResultPtr;
typedef services::SharedPtr<DistributedPartialResultStep2> DistributedPartialResultStep2Ptr;

} // namespace interface1
using interface1::Parameter;
using interface1::DistributedParameter;
using interface1::Input;
using interface1::DistributedInput;
using interface1::PartialResultBase;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep2Ptr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace init
/** @} */
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
