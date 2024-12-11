/* file: covariance_online.h */
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
//  Implementation of the interface for the correlation or variance-covariance
//  matrix algorithm in the online processing mode
//--
*/

#ifndef __COVARIANCE_ONLINE_H__
#define __COVARIANCE_ONLINE_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_types.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
/**
 * @defgroup covariance_online Online
 * @ingroup covariance
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINERIFACE"></a>
* \brief Class that spcifies interfaces of implementations of the correlation or variance-covariance matrix algorithm.
*        This class is associated with daal::algorithms::covariance::OnlineImpl class
*
* \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
* \tparam method           Computation method of the algorithm, \ref daal::algorithms::covariance::Method
*/
class OnlineContainerIface : public daal::algorithms::AnalysisContainerIface<online>
{
public:
    OnlineContainerIface() {}
    virtual ~OnlineContainerIface() {}

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() = 0;

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm.
 *        This class is associated with daal::algorithms::covariance::Online class
 *
 * \tparam method           Computation method for correlation or variance-covariance matrix, \ref daal::algorithms::covariance::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_DEFAULTDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using default computation method.
 *        This class is associated with daal::algorithms::covariance::Online class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, defaultDense, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_SINGLEPASSDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method.
 *        This class is associated with daal::algorithms::covariance::Online class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, singlePassDense, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_SUMDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using sum computation method.
 *        This class is associated with daal::algorithms::covariance::Online class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, sumDense, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_FASTCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using fast computation method that works with Compressed Sparse Rows (CSR) numeric tables.
 *        This class is associated with daal::algorithms::covariance::Online class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, fastCSR, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_SINGLEPASSCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method that works with Compressed Sparse Rows (CSR) numeric tables.
 *        This class is associated with daal::algorithms::covariance::Online class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, singlePassCSR, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINECONTAINER_ALGORITHMFPTYPE_SUMCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method that works with Compressed Sparse Rows (CSR) numeric tables.
 *        This class is associated with daal::algorithms::covariance::Online class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, sumCSR, cpu> : public OnlineContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the online processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINEIFACE"></a>
 * \brief Abstract class that specifies interface of the algorithms
 *        for computing correlation or variance-covariance matrix in the online processing mode
 */
class DAAL_EXPORT OnlineImpl : public daal::algorithms::Analysis<online>
{
public:
    typedef algorithms::covariance::Input InputType;
    typedef algorithms::covariance::OnlineParameter ParameterType;
    typedef algorithms::covariance::Result ResultType;
    typedef algorithms::covariance::PartialResult PartialResultType;

    /** Default constructor */
    OnlineImpl() { initialize(); }

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    OnlineImpl(const OnlineImpl & other) : input(other.input), parameter(other.parameter) { initialize(); }
    OnlineImpl & operator=(const OnlineImpl & other)
    {
        input     = other.input;
        parameter = other.parameter;
        initialize();
        return *this;
    }

    virtual ~OnlineImpl() {}

    /**
     * Returns the structure that contains final results of the correlation or variance-covariance matrix algorithm
     * \return Structure that contains the final results
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store final results of the correlation or variance-covariance matrix algorithm
     * \param[in] result    Structure to store the results
     */
    virtual services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the correlation or variance-covariance matrix algorithm
     * \return Structure that contains partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix algorithm
     * \param[in] partialResult    Structure to store partial results
     * \param[in] initFlag        Flag that specifies whether the partial results are initialized
     */
    virtual services::Status setPartialResult(const PartialResultPtr & partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * with a copy of input objects and parameters of this algorithm for correlation or variance-covariance
     * matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<OnlineImpl> clone() const { return services::SharedPtr<OnlineImpl>(cloneImpl()); }

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Parameter structure */

protected:
    void initialize()
    {
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResult());
    }

    virtual OnlineImpl * cloneImpl() const DAAL_C11_OVERRIDE = 0;

    PartialResultPtr _partialResult;
    ResultPtr _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINE"></a>
 * \brief Computes correlation or variance-covariance matrix in the online processing mode.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrix algorithm description and usage models</a> -->
 *
 * \tparam method           Computation method for correlation or variance-covariance matrix, \ref daal::algorithms::covariance::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 *
 * \par Enumerations
 *      - \ref Method           Correlation or variance-covariance matrix computation methods
 *      - \ref InputId          Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref PartialResultId  Identifiers of partial results of the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId         Identifiers of results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - Parameter class
 *      - PartialResult class
 *      - Result class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Online : public OnlineImpl
{
public:
    typedef OnlineImpl super;

    typedef typename super::InputType InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;
    typedef typename super::PartialResultType PartialResultType;

    /** Default constructor */
    Online() { initialize(); }

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, method> & other) : OnlineImpl(other) { initialize(); }

    virtual ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * with a copy of input objects and parameters of this algorithm for correlation or variance-covariance
     * matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, method> > clone() const { return services::SharedPtr<Online<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Online<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Online<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_partialResult.get(), _par, (int)method);
        _res               = _result.get();
        _pres              = _partialResult.get();
        return services::Status();
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return services::Status();
    }

    void initialize() { this->_ac = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, method)(&_env); }
};
/** @} */
} // namespace interface1
using interface1::OnlineContainerIface;
using interface1::OnlineContainer;
using interface1::OnlineImpl;
using interface1::Online;

} // namespace covariance
} // namespace algorithms
} // namespace daal
#endif
