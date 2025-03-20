/* file: covariance_batch.h */
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
//  matrix algorithm in the batch processing mode
//--
*/

#ifndef __COVARIANCE_BATCH_H__
#define __COVARIANCE_BATCH_H__

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
 * @defgroup covariance_batch Batch
 * @ingroup covariance
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINERIFACE"></a>
 * \brief Class that specifies interfaces of implementations of the correlation or variance-covariance matrix container.
 *        This class is associated with daal::algorithms::covariance::BatchContainerIface class.
 */
class BatchContainerIface : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /** Default constructor */
    BatchContainerIface() {}
    /** Default destructor */
    virtual ~BatchContainerIface() {}

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm.
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::covariance::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_DEFAULTDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using default computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, defaultDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SINGLEPASSDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, singlePassDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SUMDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, sumDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_FASTCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using fast computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, fastCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SINGLEPASSCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, singlePassCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SUMCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, sumCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHIFACE"></a>
 * \brief Abstract class that specifies interface of the algorithms
 *        for computing correlation or variance-covariance matrix in the batch processing mode
 */
class DAAL_EXPORT BatchImpl : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::covariance::Input InputType;
    typedef algorithms::covariance::Parameter ParameterType;
    typedef algorithms::covariance::Result ResultType;

    /** Default constructor */
    BatchImpl() : daal::algorithms::Analysis<batch>() { initialize(); }

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    BatchImpl(const BatchImpl & other) : input(other.input), parameter(other.parameter)
    {
        initialize();
        _hpar = other.daal::algorithms::Analysis<batch>::_hpar;
    }

    /**
     * Copy-assignment operator for an algorithm for correlation or variance-covariance matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    BatchImpl & operator=(const BatchImpl & other)
    {
        input     = other.input;
        parameter = other.parameter;
        initialize();
        _hpar = other.daal::algorithms::Analysis<batch>::_hpar;
        return *this;
    }

    /**
     * Returns the structure that contains correlation or variance-covariance matrix
     * \return Structure that contains the computed matrix
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of computation of the correlation or variance-covariance matrix
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
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * with a copy of input objects and parameters of this algorithm for correlation or variance-covariance
     * matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<BatchImpl> clone() const { return services::SharedPtr<BatchImpl>(cloneImpl()); }

    virtual ~BatchImpl() {}

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Parameter structure */

protected:
    ResultPtr _result;

    void initialize()
    {
        _result.reset(new ResultType());
        _in   = &input;
        _par  = &parameter;
        _hpar = nullptr;
    }
    virtual BatchImpl * cloneImpl() const DAAL_C11_OVERRIDE = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCH"></a>
 * \brief Computes correlation or variance-covariance matrix in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrix algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId  Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId Identifiers of results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - Parameter class
 *      - Result class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public BatchImpl
{
public:
    typedef BatchImpl super;

    typedef typename super::InputType InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;

    /** Default constructor */
    Batch();

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

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
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize() { this->_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env); }
};
/** @} */
} // namespace interface1
using interface1::BatchContainerIface;
using interface1::BatchContainer;
using interface1::BatchImpl;
using interface1::Batch;

} // namespace covariance
} // namespace algorithms
} // namespace daal
#endif // __COVARIANCE_BATCH_H__
