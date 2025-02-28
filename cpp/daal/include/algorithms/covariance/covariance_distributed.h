/* file: covariance_distributed.h */
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
//  matrix algorithm in the distributed processing mode
//--
*/

#ifndef __COVARIANCE_DISTRIBUTED_H__
#define __COVARIANCE_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_types.h"
#include "algorithms/covariance/covariance_online.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
/**
 * @defgroup covariance_distributed Distributed
 * @ingroup covariance
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINERIFACE"></a>
* \brief Class that spcifies interfaces of the correlation or variance-covariance matrix algorithm.
*        This class is associated with daal::algorithms::covariance::DistributedIface class
*
* \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
* \tparam method           Computation method of the algorithm, \ref daal::algorithms::covariance::Method
*/
template <ComputeStep step>
class DistributedContainerIface
{};

/**
* <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINERIFACE_STEP2MASTER"></a>
* \brief Class that spcifies interfaces of the correlation or variance-covariance matrix algorithm on master node.
*        This class is associated with daal::algorithms::covariance::DistributedIface class
*
* \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
* \tparam method           Computation method of the algorithm, \ref daal::algorithms::covariance::Method
*/
template <>
class DistributedContainerIface<step2Master> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    DistributedContainerIface() {}
    virtual ~DistributedContainerIface() {}

    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() = 0;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix  algorithm in the distributed processing mode.
 *        This class is associated with daal::algorithms::covariance::Distributed class
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm using default computation method in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, defaultDense, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm using single-pass computation method in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, singlePassDense, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm using sum computation method in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, sumDense, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using fast computation method that works with Compressed Sparse Rows (CSR) numeric tables in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, fastCSR, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method that works with Compressed Sparse Rows (CSR) numeric tables in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, singlePassCSR, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method that works with Compressed Sparse Rows (CSR) numeric tables in the distributed processing mode on master node.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, sumCSR, cpu> : public DistributedContainerIface<step2Master>
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDIFACE"></a>
 * \brief Interface for the correlation or variance-covariance matrix algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method               Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId              Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId        Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId      Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId             Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <ComputeStep step>
class DistributedIface : public daal::algorithms::Analysis<distributed>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDIFACE_STEP1LOCAL"></a>
 * \brief Interface for correlation or variance-covariance matrix computation algorithms in the distributed processing mode on local nodes.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method               Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId              Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId        Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId      Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId             Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <>
class DistributedIface<step1Local> : public OnlineImpl
{
public:
    typedef OnlineImpl super;

    typedef algorithms::covariance::DistributedInput<step1Local> InputType;
    typedef super::ParameterType ParameterType;
    typedef super::ResultType ResultType;
    typedef super::PartialResultType PartialResultType;

    /** Default constructor */
    DistributedIface() : OnlineImpl() {}

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on local node by copying input objects and parameters
     * of another algorithm for correlation or variance-covariance matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DistributedIface(const DistributedIface<step1Local> & other) : OnlineImpl(other) {}

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on local node with a copy of input objects and parameters of this algorithm
     * for correlation or variance-covariance matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<DistributedIface<step1Local> > clone() const { return services::SharedPtr<DistributedIface<step1Local> >(cloneImpl()); }

protected:
    virtual DistributedIface<step1Local> * cloneImpl() const DAAL_C11_OVERRIDE = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDIFACE_STEP2MASTER"></a>
 * \brief Interface for correlation or variance-covariance matrix computation algorithms in the distributed processing mode on master node.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method               Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId              Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId        Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId      Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId             Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <>
class DistributedIface<step2Master> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::covariance::DistributedInput<step2Master> InputType;
    typedef algorithms::covariance::Parameter ParameterType;
    typedef algorithms::covariance::Result ResultType;
    typedef algorithms::covariance::PartialResult PartialResultType;

    /** Default constructor */
    DistributedIface() { initialize(); }

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on master node by copying input objects and parameters
     * of another algorithm for correlation or variance-covariance matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DistributedIface(const DistributedIface<step2Master> & other) : parameter(other.parameter)
    {
        initialize();
        data_management::DataCollectionPtr collection = other.input.get(partialResults);
        for (size_t i = 0; i < collection->size(); i++)
        {
            input.add(partialResults, services::staticPointerCast<PartialResultType, data_management::SerializationIface>((*collection)[i]));
        }
    }

    virtual ~DistributedIface() {}

    /**
     * Returns the structure that contains final results of the correlation or variance-covariance matrix algorithm
     * \return Structure that contains final results
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
     * Returns the structure that contains computed partial results of the correlation or variance-covariance matrix algorithm
     * \return Structure that contains partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store partial results of the correlation or variance-covariance matrix algorithm
     * \param[in] partialResult    Structure to store partial results
     * \param[in] initFlag         Flag that specifies whether the partial results are initialized
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
     * Validates parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        services::Status s;
        if (this->_partialResult)
        {
            s |= this->_partialResult->check(this->_par, getMethod());
            if (!s) return s;
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }

        if (this->_result)
        {
            s |= this->_result->check(this->_pres, this->_par, getMethod());
            if (!s) return s;
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }
        return s;
    }

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on master node with a copy of input objects and parameters of this algorithm
     * for correlation or variance-covariance matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<DistributedIface<step2Master> > clone() const { return services::SharedPtr<DistributedIface<step2Master> >(cloneImpl()); }

    DistributedInput<step2Master> input; /*!< Input data structure */
    ParameterType parameter;             /*!< Parameters of the algorithm */

protected:
    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }

    virtual DistributedIface<step2Master> * cloneImpl() const DAAL_C11_OVERRIDE = 0;

    PartialResultPtr _partialResult;
    ResultPtr _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTED"></a>
 * \brief Computes correlation or variance-covariance matrix in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId          Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId    Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId  Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId         Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed : public DistributedIface<step>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes correlation or variance-covariance matrix in the first step of the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId          Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId    Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId  Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId         Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    typedef Online<algorithmFPType, method> super;

    typedef algorithms::covariance::DistributedInput<step1Local> InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;
    typedef typename super::PartialResultType PartialResultType;

    Distributed() : Online<algorithmFPType, method>() {}

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on local node by copying input objects and parameters
     * of another algorithm for correlation or variance-covariance matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other) : Online<algorithmFPType, method>(other) {}

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on local node with a copy of input objects and parameters of this algorithm
     * for correlation or variance-covariance matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes correlation or variance-covariance matrix in the second step of the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation and variance-covariance matrices  algorithm description and usage models</a> -->
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method, \ref daal::algorithms::covariance::Method
 *
 * \par Enumerations
 *      - \ref Method           Computation methods for correlation or variance-covariance matrix
 *      - \ref InputId          Identifiers of input objects for the correlation or variance-covariance matrix algorithm
 *      - \ref MasterInputId    Identifiers of input objects for the correlation or variance-covariance matrix algorithm on master node
 *      - \ref PartialResultId  Identifiers of partial results for the correlation or variance-covariance matrix algorithm
 *      - \ref ResultId         Identifiers of final results of the correlation or variance-covariance matrix algorithm
 *
 * \par References
 *      - Input class
 *      - PartialResult class
 *      - Result class
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Master, algorithmFPType, method> : public DistributedIface<step2Master>
{
public:
    typedef DistributedIface<step2Master> super;

    typedef typename super::InputType InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;
    typedef typename super::PartialResultType PartialResultType;

    /** Default constructor */
    Distributed();

    /**
     * Constructs an algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on master node by copying input objects and parameters
     * of another algorithm for correlation or variance-covariance matrix computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> & other);

    virtual ~Distributed() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated algorithm for correlation or variance-covariance matrix computation
     * in the distributed processing mode on master node with a copy of input objects and parameters of this algorithm
     * for correlation or variance-covariance matrix computation
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr result   = this->getResult();
        services::Status s = result->template allocate<algorithmFPType>(this->_partialResult.get(), this->_par, (int)method);
        this->_res         = this->_result.get();
        this->_pres        = this->_partialResult.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_partialResult->template allocate<algorithmFPType>(&(this->input), this->_par, (int)method);
        this->_pres        = this->_partialResult.get();
        return s;
    }

    void initialize() { this->_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env); }
};
/** @} */
} // namespace interface1
using interface1::DistributedContainerIface;
using interface1::DistributedContainer;
using interface1::DistributedIface;
using interface1::Distributed;

} // namespace covariance
} // namespace algorithms
} // namespace daal
#endif // __COVARIANCE_DISTRIBUTED_H__
