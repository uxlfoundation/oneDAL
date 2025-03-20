/* file: svd_types.h */
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
//  Definition of the SVD common types.
//--
*/

#ifndef __SVD_TYPES_H__
#define __SVD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svd Singular Value Decomposition
 * \copydoc daal::algorithms::svd
 * @ingroup analysis
 * @{
 */
/** \brief Contains classes to run the singular-value decomposition (SVD) algorithm */
namespace svd
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__METHOD"></a>
 * Available methods to compute results of the SVD algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__SVDRESULTFORMAT"></a>
 * Available options to return result matrices
 */
enum SVDResultFormat
{
    notRequired,         /*!< Matrix is not required */
    requiredInPackedForm /*!< Matrix in the packed format is required */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__INPUTID"></a>
 * \brief Available types of input objects for the SVD algorithm
 */
enum InputId
{
    data, /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__RESULTID"></a>
 * \brief Available types of results of the SVD algorithm
 */
enum ResultId
{
    singularValues,      /*!< Singular values         */
    leftSingularMatrix,  /*!< Left orthogonal matrix  */
    rightSingularMatrix, /*!< Right orthogonal matrix */
    lastResultId = rightSingularMatrix
};

/**
 * <a name="DAAL-ENUM-SVD__PARTIALRESULTID"></a>
 * \brief Available types of partial results of the SVD algorithm obtained in the online processing mode and in the first step in the
 * distributed processing mode
 */
enum PartialResultId
{
    outputOfStep1ForStep3, /*!< DataCollection with data computed in the first step to be transferred to the third step in the distributed
                                    * processing mode */
    outputOfStep1ForStep2, /*!< DataCollection with data computed in the first step to be transferred to the second step in the distributed
                                    * processing mode  */
    lastPartialResultId = outputOfStep1ForStep2
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTCOLLECTIONID"></a>
 * \brief Available types of partial results obtained in the second step of the SVD algorithm in the distributed processing mode, stored in the
 * DataCollection object
 */
enum DistributedPartialResultCollectionId
{
    outputOfStep2ForStep3, /*!< DataCollection with data to be transferred to the third step in the distributed processing mode */
    lastDistributedPartialResultCollectionId = outputOfStep2ForStep3
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTID"></a>
 * \brief Available types of partial results obtained in the second step of the SVD algorithm in the distributed processing mode, stored in the
 *  Result object
 */
enum DistributedPartialResultId
{
    finalResultFromStep2Master =
        lastDistributedPartialResultCollectionId + 1, /*!< Result object with singular values and the right orthogonal matrix */
    lastDistributedPartialResultId = finalResultFromStep2Master
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * \brief Available types of partial results obtained in the third step of the SVD algorithm in the distributed processing mode, stored in the
 * Result object
 */
enum DistributedPartialResultStep3Id
{
    finalResultFromStep3, /*!< Result object with singular values and the left orthogonal matrix */
    lastDistributedPartialResultStep3Id = finalResultFromStep3
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__MASTERINPUTID"></a>
 * \brief Partial results from previous steps in the distributed processing mode, required by the second step
 */
enum MasterInputId
{
    inputOfStep2FromStep1, /*!< DataCollection with data transferred from the first step to the second step in the distributed processing mode*/
    lastMasterInputId = inputOfStep2FromStep1
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVD__FINALIZEONLOCALINPUTID"></a>
 * \brief Partial results from previous steps in the distributed processing mode, required by the third step
 */
enum FinalizeOnLocalInputId
{
    inputOfStep3FromStep1, /*!< DataCollection with data transferred from the first step to the third step in the distributed processing mode */
    inputOfStep3FromStep2, /*!< DataCollection with data transferred from the second step to the third step in the distributed processing mode */
    lastFinalizeOnLocalInputId = inputOfStep3FromStep2
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVD__PARAMETER"></a>
 * \brief Parameters for the computation method of the SVD algorithm
 */
struct Parameter : public daal::algorithms::Parameter
{
    /**
     *  Default constructor
     *  \param[in] _leftSingularMatrix  Format of the matrix of left singular vectors
     *  \param[in] _rightSingularMatrix Format of the matrix of right singular vectors
     */
    Parameter(SVDResultFormat _leftSingularMatrix = requiredInPackedForm, SVDResultFormat _rightSingularMatrix = requiredInPackedForm)
        : leftSingularMatrix(_leftSingularMatrix), rightSingularMatrix(_rightSingularMatrix)
    {}

    SVDResultFormat leftSingularMatrix;  /*!< Format of the matrix of left singular vectors  >*/
    SVDResultFormat rightSingularMatrix; /*!< Format of the matrix of right singular vectors >*/
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__INPUT"></a>
 * \brief Input objects for the SVD algorithm in the batch processing and online processing modes, and the first step in the distributed
 * processing mode
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
     * Returns an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the new input object value
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    services::Status getNumberOfColumns(size_t * nFeatures) const;

    services::Status getNumberOfRows(size_t * nRows) const;

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP2INPUT"></a>
 * \brief %Input objects for the second step of  the SVD algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedStep2Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep2Input();

    /** Copy constructor */
    DistributedStep2Input(const DistributedStep2Input & other);

    /**
     * Sets an input object for the SVD algorithm
     * \param[in] id   Identifier of the input object
     * \param[in] ptr  Input object that corresponds to the given identifier
     */
    void set(MasterInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Returns an input object for the SVD algorithm
     * \param[in] id   Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(MasterInputId id) const;

    /**
     * Adds the value to KeyValueDataCollection of the input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the new input object value
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
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /**
    * Returns the number of columns in the input data set
    * \return Number of columns in the input data set
    */
    services::Status getNumberOfColumns(size_t & nCols) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3INPUT"></a>
 * \brief %Input objects for the third step of the SVD algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedStep3Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep3Input();

    /** Copy constructor */
    DistributedStep3Input(const DistributedStep3Input & other);

    /**
     * Returns an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(FinalizeOnLocalInputId id) const;

    /**
     * Sets an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the new input object value
     */
    void set(FinalizeOnLocalInputId id, const data_management::DataCollectionPtr & value);

    services::Status getSizes(size_t & nFeatures, size_t & nVectors) const;

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINEPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of  the SVD algorithm in the online processing mode or
 * the first step in the distributed processing mode
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
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Initializes additional memory to store partial results of the SVD algorithm for each subsequent compute() method
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     * \return Status of initialization
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates additional memory to store partial results of the SVD algorithm for each subsequent compute() method
     * \tparam     algorithmFPType    Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  m    Number of columns in the input data set
     * \param[in]  n    Number of rows in the input data set
     * \param[in]  par  Reference to the object with the algorithm parameters
     */
    template <typename algorithmFPType>
    services::Status addPartialResultStorage(size_t m, size_t n, Parameter & par);

    /**
     * Returns partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(PartialResultId id) const;

    /**
     * Sets partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(PartialResultId id, const data_management::DataCollectionPtr & value);

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    size_t getNumberOfColumns() const;

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
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the SVD algorithm in the batch processing mode
 *        or with the finalizeCompute() method in the online processing mode or steps 2 and 3 in the distributed processing mode
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
     * Returns a result of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] partialResult  Pointer to the partial result
     * \param[in] parameter      Pointer to the parameter
     * \param[in] method         Algorithm computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
     * Sets the final result of the SVD algorithm
     * \param[in] id    Identifier of the final result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
      * Checks final results of the algorithm
      * \param[in] input  Pointer to input objects
      * \param[in] par    Pointer to parameters
      * \param[in] method Computation method
      * \return Errors detected while checking
      */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks the result parameter of the SVD algorithm
     * \param[in] pres    Partial result of the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    services::Status check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \tparam     algorithmFPType  Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    services::Status allocateImpl(size_t m, size_t n);

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
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm in the second step in the
 * distributed processing mode
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
     * Allocates memory to store partial results of the SVD algorithm
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory to store partial results of the SVD algorithm based on the known structure of partial results from step 1 in the
     * distributed processing mode.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \tparam     algorithmFPType Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  inCollection    KeyValueDataCollection of all partial results from the first step of  the SVD algorithm in the distributed
     *                             processing mode
     * \param[out] nBlocks         Number of rows in the input data set
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status setPartialResultStorage(data_management::KeyValueDataCollection * inCollection, size_t & nBlocks);

    /**
     * Returns partial results of the SVD algorithm.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedPartialResultCollectionId id) const;

    /**
     * Returns the DataCollection from outputOfStep2ForStep3 results of the SVD algorithm.
     * \param[in] id    Identifier of the partial result
     * \param[in] idx   Index of the DataCollection within KeyValueDataCollcetion of the partial result
     * \return          Value that corresponds to the given identifier and index
     */
    data_management::DataCollectionPtr get(DistributedPartialResultCollectionId id, size_t idx) const;

    /**
     * Returns results of the SVD algorithm with singular values and the left orthogonal matrix calculated
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    ResultPtr get(DistributedPartialResultId id) const;

    /**
     * Sets KeyValueDataCollection to store partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(DistributedPartialResultCollectionId id, const data_management::KeyValueDataCollectionPtr & value);

    /**
     * Sets the Result object to store results of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultId id, const ResultPtr & value);

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks final results of the algorithm
     * \param[in] input     Pointer to input objects
     * \param[in] parameter Pointer to parameters
     * \param[in] method    Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

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
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm
 *        in the third step in the distributed processing mode
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
     * Allocates memory to store partial results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory to store partial results of the SVD algorithm obtained in the third step in the distributed processing mode
     * \tparam     algorithmFPType            Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  qCollection  DataCollection of all partial results from step 1 of the SVD algorithm in the distributed processing mode
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status setPartialResultStorage(data_management::DataCollection * qCollection);

    /**
     * Returns results of the SVD algorithm with singular values and the left orthogonal matrix calculated
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    ResultPtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets the Result object to store results of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultStep3Id id, const ResultPtr & value);

    /**
     * Checks partial results of the algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     * \return Errors detected while checking
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep3> DistributedPartialResultStep3Ptr;
/** @} */
} // namespace interface1
using interface1::Parameter;
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

} // namespace svd
} // namespace algorithms
} // namespace daal
#endif
