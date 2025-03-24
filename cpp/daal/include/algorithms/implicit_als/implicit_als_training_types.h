/* file: implicit_als_training_types.h */
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

#ifndef __IMPLICIT_ALS_TRAINING_TYPES_H__
#define __IMPLICIT_ALS_TRAINING_TYPES_H__

#include "algorithms/implicit_als/implicit_als_model.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the implicit ALS algorithm
 */
namespace implicit_als
{
/**
 * @defgroup implicit_als_training Training
 * \copydoc daal::algorithms::implicit_als::training
 * @ingroup implicit_als
 * @{
 */
/**
 * \brief Contains classes of the implicit ALS training algorithm
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__METHOD"></a>
 * Available methods for training the implicit ALS model
 */
enum Method
{
    defaultDense = 0, /*!< Default: method proposed by Hu, Koren, Volinsky for input data stored in the dense format */
    fastCSR      = 1  /*!< Method proposed by Hu, Koren, Volinsky for input data stored in the compressed sparse row (CSR) format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__NUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm
 */
enum NumericTableInputId
{
    data, /*!< Input data table that contains ratings */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__MODELINPUTID"></a>
 * Available identifiers of input model objects for the implicit ALS training algorithm
 */
enum ModelInputId
{
    inputModel       = lastNumericTableInputId + 1, /*!< Initial model that contains initialized factors */
    lastModelInputId = inputModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__PARTIALMODELINPUTID"></a>
 * Available identifiers of input partial model objects of the implicit ALS training algorithm
 */
enum PartialModelInputId
{
    partialModel, /*!< Partial model that contains factors obtained
                                         in the previous step of the distributed processing mode */
    lastPartialModelInputId = partialModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__MASTERINPUTID"></a>
 * Partial results obtained in the previous step and required by the second step
 * of the distributed processing mode
 */
enum MasterInputId
{
    inputOfStep2FromStep1, /*!< Partial results of the implicit ALS training algorithm computed in the first step
                                         and to be transferred to the second step of the distributed processing mode */
    lastMasterInputId = inputOfStep2FromStep1
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    outputOfStep1ForStep2, /*!< Partial results of the implicit ALS training algorithm computed in the first step
                                         and to be transferred to the second step of the distributed processing mode */
    lastDistributedPartialResultStep1Id = outputOfStep1ForStep2
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    outputOfStep2ForStep4, /*!< Partial results of the implicit ALS training algorithm computed in the second step
                                         and to be transferred to the fourth step of the distributed processing mode */
    lastDistributedPartialResultStep2Id = outputOfStep2ForStep4
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP3LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum Step3LocalCollectionInputId
{
    partialModelBlocksToNode        = lastDistributedPartialResultStep2Id + 1, /*!< \DAAL_DEPRECATED_USE{ inputOfStep3FromInit }
                                         Partial results of the implicit ALS initialization algorithm to be transferred
                                         to the third step of the implicit ALS training algorithm in the distributed processing mode */
    inputOfStep3FromInit            = partialModelBlocksToNode, /*!< Partial results of the implicit ALS initialization algorithm to be transferred
                                                        to the third step of the implicit ALS training algorithm in the distributed processing mode */
    lastStep3LocalCollectionInputId = inputOfStep3FromInit
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP3LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum Step3LocalNumericTableInputId
{
    offset = lastStep3LocalCollectionInputId + 1, /*!< Pointer to the 1x1 numeric table that holds the global index of the starting row
                                                   of the input partial model */
    lastStep3LocalNumericTableInputId = offset
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    outputOfStep3ForStep4, /*!< Partial results of the implicit ALS training algorithm computed in the third step
                                         and to be transferred to the fourth step of the distributed processing mode */
    lastDistributedPartialResultStep3Id = outputOfStep3ForStep4
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP4LOCALPARTIALMODELSINPUTID"></a>
 * Available identifiers of input key-value data collection objects for the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum Step4LocalPartialModelsInputId
{
    partialModels, /*!< Key-value data collection that contains partial models consisting of user factors/item factors
                                         computed in the third step of the distributed processing mode.
                                         Each element of the collection contains an object of the PartialModel class. */
    lastStep4LocalPartialModelsInputId = partialModels
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP4LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum Step4LocalNumericTableInputId
{
    partialData = lastStep4LocalPartialModelsInputId
                  + 1,     /*!< Pointer to the CSR numeric table that holds a block of either users or items from the input data set */
    inputOfStep4FromStep2, /*!< Pointer to the nFactors x nFactors numeric table computed in the second step
                                                          of the distributed processing mode */
    lastStep4LocalNumericTableInputId = inputOfStep4FromStep2
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__PARTIALRESULTID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep4Id
{
    outputOfStep4ForStep1,                         /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                         and to be transferred to the first step of the distributed processing mode */
    outputOfStep4ForStep3 = outputOfStep4ForStep1, /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                                        and to be transferred to the third step of the distributed processing mode */
    outputOfStep4         = outputOfStep4ForStep3, /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                                        and to be used in implicit ALS PartialModel-based prediction */
    lastDistributedPartialResultStep4Id = outputOfStep4
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the results of the implicit ALS training algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other) : daal::algorithms::Input(other) {}

    virtual ~Input() {}

    /**
     * Returns the input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns the input initial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(ModelInputId id) const;

    /**
     * Sets the input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input initial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const ModelPtr & ptr);

    /**
     * Returns the number of users equal to the number of rows in the input data set
     * \return Number of users
     */
    size_t getNumberOfUsers() const;

    /**
     * Returns the number of items equal to the number of columns in the input data set
     * \return Number of items
     */
    size_t getNumberOfItems() const;

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the first step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step1Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    PartialModelPtr get(PartialModelInputId id) const;

    /**
     * Sets an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const PartialModelPtr & ptr);

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the first step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep1)
    /** Default constructor */
    DistributedPartialResultStep1();

    virtual ~DistributedPartialResultStep1() {}

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep1Id id) const;

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep1Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the implicit ALS algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
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
};
typedef services::SharedPtr<DistributedPartialResultStep1> DistributedPartialResultStep1Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the second step of the
 * distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step2Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(MasterInputId id) const;

    /**
     * Sets an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(MasterInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the implicit ALS training algorithm in the second step
     * of the distributed processing mode
     * \param[in] id            Identifier of the input object
     * \param[in] partialResult Pointer to the partial result obtained in the previous step of the distributed processing mode
     */
    void add(MasterInputId id, const DistributedPartialResultStep1Ptr & partialResult);

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm in the second step
     * of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep2)
    /** Default constructor */
    DistributedPartialResultStep2();

    virtual ~DistributedPartialResultStep2() {}

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
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
};
typedef services::SharedPtr<DistributedPartialResultStep2> DistributedPartialResultStep2Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT_STEP3LOCAL"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the third step of
 * the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step3Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input partial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    PartialModelPtr get(PartialModelInputId id) const;

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(Step3LocalCollectionInputId id) const;

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step3LocalNumericTableInputId id) const;

    /**
     * Sets an input partial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const PartialModelPtr & ptr);

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalCollectionInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the number of blocks of data used in distributed computations
     * \return Number of blocks of data
     */
    size_t getNumberOfBlocks() const;

    /**
     * Returns the index of the starting row of the input partial model
     * \return Index of the starting row of the input partial model
     */
    size_t getOffset() const;

    /**
     * Returns the numeric table that contains the indices of factors that should be transferred to a specified node
     * \param[in] key Index of the node
     * \return Numeric table that contains the indices of factors that should be transferred to a specified node
     */
    data_management::NumericTablePtr getOutBlockIndices(size_t key) const;

    /**
     * Checks the parameters and input objects of the implicit ALS training algorithm in the first step of
     * the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the the third step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep3)
    /** Default constructor */
    DistributedPartialResultStep3();

    virtual ~DistributedPartialResultStep3() {}

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the implicit ALS training algorithm
     *
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Returns a partial model obtained with the compute() method of the implicit ALS algorithm in the third step of the
     * distributed processing mode
     *
     * \param[in] id    Identifier of the partial result
     * \param[in] key   Index of the partial model in the key-value data collection
     * \return          Pointer to the partial model object
     */
    PartialModelPtr get(DistributedPartialResultStep3Id id, size_t key) const;

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedPartialResultStep3Id id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
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
};
typedef services::SharedPtr<DistributedPartialResultStep3> DistributedPartialResultStep3Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT_STEP4LOCAL"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the fourth step of
 * the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step4Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     *
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier.
     *                  A key-value data collection contains partial models consisting of user factors/item factors
     *                  computed in the third step of the distributed processing mode
     */
    data_management::KeyValueDataCollectionPtr get(Step4LocalPartialModelsInputId id) const;

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step4LocalNumericTableInputId id) const;

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalPartialModelsInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the number of rows in the partial matrix of users factors/items factors
     * \return Number of rows in the partial matrix of factors
     */
    size_t getNumberOfRows() const;

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm in the first step of
     * the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the the fourth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep4 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep4)
    /** Default constructor */
    DistributedPartialResultStep4();

    virtual ~DistributedPartialResultStep4() {}

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    PartialModelPtr get(DistributedPartialResultStep4Id id) const;

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(DistributedPartialResultStep4Id id, const PartialModelPtr & ptr);

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
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
};
typedef services::SharedPtr<DistributedPartialResultStep4> DistributedPartialResultStep4Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__RESULT"></a>
 * \brief Provides methods to access the results obtained with the compute() method of the implicit ALS training algorithm
 * in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    /** Default constructor */
    Result();

    /**
     * Allocates memory to store the results of the implicit ALS training algorithm
     * \param[in] input         Pointer to the input structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns the result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    daal::algorithms::implicit_als::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const daal::algorithms::implicit_als::ModelPtr & ptr);

    /**
     * Checks the result of the implicit ALS training algorithm
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
} // namespace interface1
using interface1::Input;
using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep1Ptr;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep2Ptr;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep3Ptr;
using interface1::DistributedPartialResultStep4;
using interface1::DistributedPartialResultStep4Ptr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
