/* file: qr_types.h */
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
//  Definition of QR common types.
//--
*/

#ifndef __QR_TYPES_H__
#define __QR_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup qr QR Decomposition
 * \copydoc daal::algorithms::qr
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup qr_without_pivoting QR Decomposition without Pivoting
 * \brief Contains classes for computing the results of the QR decomposition algorithm without Pivoting
 * @ingroup qr
 * @{
 */
/** \brief Contains classes for computing the results of the QR decomposition algorithm */
namespace qr
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__METHOD"></a>
 * Available methods for computing the QR decomposition algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__INPUTID"></a>
 * Available types of input objects for the QR decomposition algorithm
 */
enum InputId
{
    data, /*!< Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__RESULTID"></a>
 * Available types of results of the QR decomposition algorithm
 */
enum ResultId
{
    matrixQ, /*!< Orthogonal Matrix Q */
    matrixR, /*!< Upper Triangular Matrix R */
    lastResultId = matrixR
};

/**
 * <a name="DAAL-ENUM-QR__PARTIALRESULTID"></a>
 * Available types of partial results of the QR decomposition algorithm in the online processing mode and of the first step of the
 * QR decomposition algorithm in the distributed processing mode
 */
enum PartialResultId
{
    outputOfStep1ForStep3, /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                    * to the third step in the distributed processing mode */
    outputOfStep1ForStep2, /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                    * to the second step in the distributed processing mode */
    lastPartialResultId = outputOfStep1ForStep2
};

/**
 * <a name="DAAL-ENUM-QR__DISTRIBUTEDPARTIALRESULTCOLLECTIONID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in DataCollection object in the distributed
 * processing mode
 */
enum DistributedPartialResultCollectionId
{
    outputOfStep2ForStep3, /*!< Partial results of the QR decomposition algorithms to be transferred  to the third step in the distributed
                                    * processing mode */
    lastDistributedPartialResultCollectionId = outputOfStep2ForStep3
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__DISTRIBUTEDPARTIALRESULTID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in Result object in the distributed processing mode
 */
enum DistributedPartialResultId
{
    finalResultFromStep2Master     = lastDistributedPartialResultCollectionId + 1, /*!< Result object with R matrix */
    lastDistributedPartialResultId = finalResultFromStep2Master
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in Result object in the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    finalResultFromStep3, /*!< Result object with Q matrix */
    lastDistributedPartialResultStep3Id = finalResultFromStep3
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__MASTERINPUTID"></a>
 * Partial results from the previous steps in the distributed processing mode required by the second distributed step of the algorithm
 */
enum MasterInputId
{
    inputOfStep2FromStep1, /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred  to the
                                  * second step in the distributed processing mode */
    lastMasterInputId = inputOfStep2FromStep1
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QR__FINALIZEONLOCALINPUTID"></a>
 * Partial results from the previous steps in the distributed processing mode required by the third distributed step
 */
enum FinalizeOnLocalInputId
{
    inputOfStep3FromStep1, /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                  * to the third step in the distributed processing mode */
    inputOfStep3FromStep2, /*!< Partial results of the QR decomposition algorithms computed on the second step and to be transferred
                                  * to the third step in the distributed processing mode */
    lastFinalizeOnLocalInputId = inputOfStep3FromStep2
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__INPUT"></a>
 * \brief Input objects for the QR decomposition algorithm in the batch and online processing modes and for the first distributed step of the
 * algorithm.
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Returns input object of the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    services::Status getNumberOfColumns(size_t * nFeatures) const;

    services::Status getNumberOfRows(size_t * nRows) const;

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP2INPUT"></a>
 * \brief Input objects for the second step of the QR decomposition algorithm in the distributed processing mode.
 */
class DAAL_EXPORT DistributedStep2Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep2Input();

    /** Copy constructor */
    DistributedStep2Input(const DistributedStep2Input & other);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    services::Status getNumberOfColumns(size_t * nFeatures) const;

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object that corresponds to the given identifier
     */
    void set(MasterInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Returns input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(MasterInputId id) const;

    /**
     * Adds input object to KeyValueDataCollection  of the QR decomposition algorithm
     * \param[in] id    Identifier of input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the input object value
     */
    void add(MasterInputId id, size_t key, const data_management::DataCollectionPtr & value);

    /**
    * Returns the number of blocks in the input data set
    * \return Number of blocks in the input data set
    */
    size_t getNBlocks();

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP3INPUT"></a>
 * \brief Input objects for the third step of the QR decomposition algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedStep3Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep3Input();

    /** Copy constructor */
    DistributedStep3Input(const DistributedStep3Input & other);

    /**
     * Returns input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(FinalizeOnLocalInputId id) const;

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object value
     */
    void set(FinalizeOnLocalInputId id, const data_management::DataCollectionPtr & value);

    services::Status getSizes(size_t & nFeatures, size_t & nVectors) const;

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__ONLINEPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the QR decomposition algorithm
 *        in the online processing mode or on the first step of the algorithm in the distributed processing mode
 */
class DAAL_EXPORT OnlinePartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(OnlinePartialResult)
    /** Default constructor */
    OnlinePartialResult();
    /** Default destructor */
    virtual ~OnlinePartialResult() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Initializes memory for storing partial results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     * \return Status of initialization
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates additional memory for storing partial results of the QR decomposition algorithm for each subsequent call to compute method
     * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status addPartialResultStorage(size_t m, size_t n);

    /**
     * Returns partial result of the QR decomposition algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(PartialResultId id) const;

    /**
    * Sets an input object for the QR decomposition algorithm
    * \param[in] id    Identifier of partial result
    * \param[in] value Pointer to the partial result
    */
    void set(PartialResultId id, const data_management::DataCollectionPtr & value);

    /**
    * Checks parameters of the algorithm
    * \param[in] input Input of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                   int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */

    size_t getNumberOfColumns() const;

    /**
    * Returns the number of rows in the input data set
    * \return Number of rows in the input data set
    */
    size_t getNumberOfRows() const;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    services::Status checkImpl(const daal::algorithms::Parameter * parameter, int method, size_t nFeatures, size_t nVectors) const;
};
typedef services::SharedPtr<OnlinePartialResult> OnlinePartialResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the QR decomposition algorithm
 *        in the batch processing mode or finalizeCompute() method of algorithm in the online processing mode
 *        or on the second and third steps of the algorithm in the distributed processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    /** Default constructor */
    Result();
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \param[in] partialResult  Pointer to partial result
     * \param[in] parameter      Pointer to the result
     * \param[in] method         Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
    * Sets an input object for the QR decomposition algorithm
    * \param[in] id    Identifier of the result
    * \param[in] value Pointer to the result
    */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
       * Checks final results of the algorithm
      * \param[in] input  Pointer to input objects
      * \param[in] par    Pointer to parameters
      * \param[in] method Computation method
      */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                   int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the result parameter of the QR algorithm
    * \param[in] pres    Partial result of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    */
    virtual services::Status check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * par,
                                   int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocateImpl(size_t m, size_t n);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the
 *        second step of the QR decomposition algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResult)
    /** Default constructor */
    DistributedPartialResult();
    /** Default destructor */
    virtual ~DistributedPartialResult() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input  Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm based on known structure of partial results from the
     * first steps of the algorithm in the distributed processing mode.
     * KeyValueDataCollection under outputOfStep2ForStep3 is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \tparam     algorithmFPType             Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  inCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
     * processing mode
     * \param[out] nBlocks  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status setPartialResultStorage(data_management::KeyValueDataCollection * inCollection, size_t & nBlocks);

    /**
     * Returns partial result of the QR decomposition algorithm.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedPartialResultCollectionId id) const;

    /**
     * Returns the result of the QR decomposition algorithm with the matrix R calculated
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    ResultPtr get(DistributedPartialResultId id) const;

    /**
     * Sets KeyValueDataCollection to store partial result of the QR decomposition algorithm
     * \param[in] id    Identifier of partial result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultCollectionId id, const data_management::KeyValueDataCollectionPtr & value);

    /**
     * Sets Result object to store the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultId id, const ResultPtr & value);

    /**
    * Checks partial results of the algorithm
    * \param[in] parameter Pointer to parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
      * Checks final results of the algorithm
      * \param[in] input      Pointer to input objects
      * \param[in] parameter  Pointer to parameters
      * \param[in] method     Computation method
      */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                   int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResult> DistributedPartialResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the third step of the QR decomposition algorithm
 *        in the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep3)
    /** Default constructor */
    DistributedPartialResultStep3();
    /** Default destructor */
    virtual ~DistributedPartialResultStep3() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory for storing partial results of the third step of the QR decomposition algorithm in the distributed processing mode
     * \tparam     algorithmFPType            Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  qCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
     * processing mode
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status setPartialResultStorage(data_management::DataCollection * qCollection);

    /**
     * Returns the result of the QR decomposition algorithm with the matrix Q calculated
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    ResultPtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets Result object to store the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultStep3Id id, const ResultPtr & value);

    /**
    * Checks partial results of the algorithm
    * \param[in] input     Pointer to the input object
    * \param[in] parameter Pointer to parameters
    * \param[in] method Computation method
    */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                   int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep3> DistributedPartialResultStep3Ptr;

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__QR__PARAMETER"></a>
 * \brief Parameters for the QR decomposition compute method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Default constructor
     */
    Parameter() {}
};
/** @} */
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::DistributedStep2Input;
using interface1::DistributedStep3Input;
using interface1::OnlinePartialResult;
using interface1::OnlinePartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;
using interface1::DistributedPartialResult;
using interface1::DistributedPartialResultPtr;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep3Ptr;
using interface1::Parameter;

} // namespace qr
} // namespace algorithms
} // namespace daal
#endif
