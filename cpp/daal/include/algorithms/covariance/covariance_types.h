/* file: covariance_types.h */
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
//  Definition of Covariance common types.
//--
*/

#ifndef __COVARIANCE_TYPES_H__
#define __COVARIANCE_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup covariance Correlation and Variance-Covariance Matrices
 * \copydoc daal::algorithms::covariance
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for computing the correlation or variance-covariance matrix
 */
namespace covariance
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__METHOD"></a>
 * Available computation methods for variance-covariance or correlation matrix
 */
enum Method
{
    defaultDense    = 0, /*!< Default: performance-oriented method. Works with all types of numeric tables */
    singlePassDense = 1, /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Works with all types of numeric tables */
    sumDense        = 2, /*!< Precomputed sum: implementation of moments computation algorithm in the case of a precomputed sum.
                                     Works with all types of numeric tables */
    fastCSR         = 3, /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
    singlePassCSR   = 4, /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Works with CSR numeric tables */
    sumCSR          = 5  /*!< Precomputed sum: implementation of the algorithm in the case of a precomputed sum.
                                     Works with CSR numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__INPUTID"></a>
 * Available identifiers of input objects for the correlation or variance-covariance matrix algorithm
 */
enum InputId
{
    data, /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__PARTIALRESULTID"></a>
 * Available identifiers of partial results of the correlation or variance-covariance matrix algorithm
 */
enum PartialResultId
{
    nObservations, /*!< Number of observations processed so far */
    crossProduct,  /*!< Cross-product matrix computed so far */
    sum,           /*!< Vector of sums computed so far */
    lastPartialResultId = sum
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__RESULTID"></a>
 * \brief Available identifiers of results of the correlation or variance-covariance matrix algorithm
 */
enum ResultId
{
    covariance,               /*!< Variance-covariance matrix */
    correlation = covariance, /*!< Correlation matrix */
    mean,                     /*!< Vector of means */
    lastResultId = mean
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__OUTPUTMATRIXTYPE"></a>
 * Available types of the computed matrix for Covariance
 */
enum OutputMatrixType
{
    covarianceMatrix, /*!< Variance-Covariance matrix */
    correlationMatrix /*!< Correlation matrix */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__MASTERINPUTID"></a>
 * \brief Available identifiers of master node input arguments of the Covariance algorithm
 */
enum MasterInputId
{
    partialResults, /*!< Collection of partial results trained on local nodes */
    lastMasterInputId = partialResults
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INPUTIFACE"></a>
 * \brief Abstract class that specifies interface for classes that declare input of the correlation or variance-covariance matrix algorithm
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements);
    InputIface(const InputIface & other);
    InputIface & operator=(const InputIface & other);
    virtual size_t getNumberOfFeatures() const = 0;
    virtual ~InputIface();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INPUT"></a>
 * \brief %Input objects of the correlation or variance-covariance matrix algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input();

    /**
     * Returns number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Returns the input object of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks algorithm parameters
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the correlation or variance-covariance matrix algorithm
 *        in the online or distributed processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    PartialResult();

    virtual ~PartialResult() {}

    /**
     * Allocates memory to store partial results of the correlation or variance-covariance matrix algorithm
     * \param[in] input     %Input objects of the algorithm
     * \param[in] parameter Parameters of the algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Initializes memory to store partial results of the correlation or variance-covariance matrix algorithm
     * \param[in] input     %Input objects of the algorithm
     * \param[in] parameter Parameters of the algorithm
     * \param[in] method    Computation method
     * \return Status of initialization
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Gets the number of columns in the partial result of the correlation or variance-covariance matrix algorithm
     * \return Number of columns in the partial result
     */
    size_t getNumberOfFeatures() const;

    /**
     * Returns the partial result of the correlation or variance-covariance matrix algorithm
     * \param[in] id   Identifier of the partial result, \ref PartialResultId
     * \return Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultId id) const;

    /**
     * Sets the partial result of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(PartialResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Check correctness of the partial result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
    * Check the correctness of PartialResult object
    * \param[in] parameter Pointer to the structure of the parameters of the algorithm
    * \param[in] method    Computation method
    */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(size_t nFeatures) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__COVARIANCE__PARAMETER"></a>
 * \brief Parameters of the correlation or variance-covariance matrix algorithm
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /** Default constructor */
    Parameter();
    bool assumeCentered = false;       /*!< Determines if mean centering is necessary or not*/
    bool bias           = false;       /*!< Determines if covariance estimation biased or not*/
    OutputMatrixType outputMatrixType; /*!< Type of the computed matrix */
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__COVARIANCE__ONLINEPARAMETER"></a>
 * \brief Parameters of the correlation or variance-covariance matrix algorithm in the online processing mode
 */
struct DAAL_EXPORT OnlineParameter : public Parameter
{
    /** Default constructor */
    OnlineParameter();

    /**
    *  Constructs parameters of the Covariance Online algorithm by copying another parameters of the Covariance Online algorithm
    *  \param[in] other    Parameters of the Covariance Online algorithm
    */
    OnlineParameter(const OnlineParameter & other);

    /**
    *  Copy-assignment operator for parameters of the Covariance Online algorithm
    *  \param[in] other    Parameters of the Covariance Online algorithm
    */
    OnlineParameter & operator=(const OnlineParameter & other);

    /**
     * Check the correctness of the %OnlineParameter object
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of
 *        the correlation or variance-covariance matrix algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {}

    /**
     * Allocates memory to store final results of the correlation or variance-covariance matrix algorithm
     * \param[in] input     %Input objects of the algorithm
     * \param[in] parameter Parameters of the algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory for storing Covariance final results
     * \param[in] partialResult      Partial Results arguments of the covariance algorithm
     * \param[in] parameter          Parameters of the covariance algorithm
     * \param[in] method             Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
     * Returns the final result of the correlation or variance-covariance matrix algorithm
     * \param[in] id   Identifier of the result, \ref ResultId
     * \return Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the correlation or variance-covariance matrix algorithm
     * \param[in] id        Identifier of the result
     * \param[in] value     Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Check correctness of the result
     * \param[in] partialResult     Pointer to the partial result arguments structure
     * \param[in] parameter         Pointer to the structure of the parameters of the algorithm
     * \param[in] method            Computation method
     */
    services::Status check(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                           int method) const DAAL_C11_OVERRIDE;

    /**
     * Check correctness of the result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(size_t nFeatures, OutputMatrixType outputMatrixType) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *
 * \tparam step             Step of the distributed computing algorithm, \ref ComputeStep
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *        Represents inputs of the algorithm on local node.
 */
template <>
class DAAL_EXPORT DistributedInput<step1Local> : public Input
{
public:
    DistributedInput() : Input() {}
    DistributedInput(const DistributedInput & other)             = default;
    DistributedInput & operator=(const DistributedInput & other) = default;

    virtual ~DistributedInput() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *        Represents inputs of the algorithm on master node.
 */
template <>
class DAAL_EXPORT DistributedInput<step2Master> : public InputIface
{
public:
    DistributedInput();
    DistributedInput(const DistributedInput & other);
    DistributedInput & operator=(const DistributedInput & other);

    virtual ~DistributedInput();

    /**
     * Returns number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Adds partial result to the end of DataCollection of input arguments of the Distributed Covariance algorithm
     * \param[in] id            Input arguments's identifier
     * \param[in] partialResult Partial result obtained on the first step of the distributed algorithm
     */
    void add(MasterInputId id, const PartialResultPtr & partialResult);

    /**
     * Returns collectionof inputs
     * \param[in] id   Partial result's identifier, \ref MasterInputId
     * \return Collection of distributed inputs
     */
    data_management::DataCollectionPtr get(MasterInputId id) const;

    /**
     * Check the correctness of DistributedInput<step2Master> object
     * \param[in] parameter Pointer to the structure of the parameters of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Parameter;
using interface1::OnlineParameter;
using interface1::Result;
using interface1::ResultPtr;
using interface1::DistributedInput;

} // namespace covariance
} // namespace algorithms
} // namespace daal
#endif // __COVARIANCE_TYPES_H__
