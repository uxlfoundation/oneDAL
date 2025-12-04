/* file: kmeans_init_distributed.h */
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
//  Implementation of the interface for initializing K-Means algorithm
//  in the distributed processing mode
//--
*/

#ifndef __KMEANS_INIT_DISTRIBITED_H__
#define __KMEANS_INIT_DISTRIBITED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_init_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface2
{
/**
 * @defgroup kmeans_init_distributed Distributed
 * @ingroup kmeans_init
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of initialization of K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::init::Distributed class
 *        and supports the method of computing initial clusters for K-Means algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref daal::algorithms::kmeans::init::Method
 *
 * \DAAL_DEPRECATED
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for computing initial clusters for K-Means algorithm in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means initialization algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of K-Means initialization algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for computing initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the 2nd step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means initialization algorithm in the 2nd step of the
     * distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of K-Means initialization algorithm in the 2nd step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode
*        performed on a local node
* \DAAL_DEPRECATED
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 2nd step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 2nd step of the
    * distributed processing mode
    */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
    * Computes the result of K-Means initialization algorithm in the 2nd step of the
    * distributed processing mode
    */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP3MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 3rd step of the distributed processing mode
*        performed on the master mode
* \DAAL_DEPRECATED
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 3rd step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 3rd step of the
    * distributed processing mode
    */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
    * Computes the result of K-Means initialization algorithm in the 3rd step of the
    * distributed processing mode
    */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for comp
* \DAAL_DEPRECATED
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step4Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 4th step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 4th step of the
    * distributed processing mode
    */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
    * Computes the result of K-Means initialization algorithm in the 4th step of the
    * distributed processing mode
    */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP5MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 5th step of the distributed processing mode
*        performed on the master node
* \DAAL_DEPRECATED
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step5Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 5th step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DAAL_DEPRECATED DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 5th step of the
    * distributed processing mode
    */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
    * Computes the result of K-Means initialization algorithm in the 5th step of the
    * distributed processing mode
    */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED"></a>
 *  \brief Base class representing K-Means algorithm initialization in the distributed processing mode
 */
class DistributedBase : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::kmeans::init::Parameter ParameterType;
    /** Default destructor */
    virtual ~DistributedBase()
    {
        if (_par)
        {
            delete _par;
        }
    }

protected:
    DistributedBase() {}

    explicit DistributedBase(ParameterType * parameter) { _par = parameter; }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED"></a>
 * \brief Computes initial clusters for K-Means algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Methods of computing initial clusters for K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
 *      - \ref ResultId Identifiers of results of computing initial clusters for K-Means algorithm
 *
 * \par References
 *      - Input  class
 *      - Result class
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes initial clusters for K-Means algorithm in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method            Method of computing initial clusters for the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Methods of computing initial clusters for K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
 *      - \ref ResultId Identifiers of results of computing initial clusters for K-Means algorithm
 *
 * \par References
 *      - Input  class
 *      - Result class
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public DistributedBase
{
public:
    typedef algorithms::kmeans::init::Input InputType;
    typedef algorithms::kmeans::init::Result ResultType;
    typedef algorithms::kmeans::init::PartialResult PartialResultType;

    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     *  \param[in] nRowsTotal  Number of rows in all data sets
     *  \param[in] offset      Offset in the total data set specifying the start of a block stored on a given local node
     */
    Distributed(size_t nClusters, size_t nRowsTotal, size_t offset = 0);
    /**
    * Copy constructor
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the results of computing initial clusters for K-Means algorithm
     * \return Structure that contains the results of computing initial clusters for K-Means algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store the results of computing initial clusters for K-Means algorithm
     * \param[in] result  Structure to store the results of computing initial clusters for K-Means algorithm
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
     * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
     */
    services::Status setPartialResult(const PartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE { return services::Status(); }

    /**
     * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
     * with a copy of input objects and parameters of this algorithm
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

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< K-Means init parameters structure */

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Methods of computing initial clusters for K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
 *      - \ref ResultId Identifiers of results of computing initial clusters for K-Means algorithm
 *
 * \par References
 *      - Input  class
 *      - Result class
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Master, algorithmFPType, method> : public DistributedBase
{
public:
    typedef algorithms::kmeans::init::DistributedStep2MasterInput InputType;
    typedef algorithms::kmeans::init::Result ResultType;
    typedef algorithms::kmeans::init::PartialResult PartialResultType;

    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     *  \param[in] offset      Offset in the total data set specifying the start of a block stored on a given local node
     */
    Distributed(size_t nClusters, size_t offset = 0);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the results of computing initial clusters for K-Means algorithm
     * \return Structure that contains the results of computing initial clusters for K-Means algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store the results of computing initial clusters for K-Means algorithm
     * \param[in] result  Structure to store the results of computing initial clusters for K-Means algorithm */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
     * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
     */
    services::Status setPartialResult(const PartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Validates the parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        services::Status s;
        if (_partialResult)
        {
            s |= _partialResult->check(_par, method);
            if (!s)
            {
                return s;
            }
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }

        if (_result)
        {
            s |= _result->check(&input, _par, method);
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }
        return s;
    }

    /**
     * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
     * with a copy of input objects and parameters of this algorithm
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
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        if (!s) return s;
        _pres = _partialResult.get();
        if (!_res)
        {
            _result.reset(new ResultType());
            s |= _result->allocate<algorithmFPType>(&input, _par, (int)method);
            _res = _result.get();
        }
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< K-Means init parameters structure */

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDPLUSPLUS"></a>
 *  \brief Base class representing K-Means algorithm initialization in the distributed processing mode
 */
class DistributedStep2LocalPlusPlusBase : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::kmeans::init::DistributedStep2LocalPlusPlusParameter ParameterType;
    /** Default destructor */
    virtual ~DistributedStep2LocalPlusPlusBase() {}

protected:
    DistributedStep2LocalPlusPlusBase() {}

    explicit DistributedStep2LocalPlusPlusBase(ParameterType * parameter) { _par = parameter; }
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP2LOCAL_ALGORITHMFPTYPE_METHOD"></a>
* \brief Computes initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode.
*        Used with plusPlus and parallelPlus methods only on a local node.
* <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
* \tparam method            Method of computing initial clusters for the algorithm, \ref Method
*
* \par Enumerations
*      - \ref Method   Methods of computing initial clusters for K-Means algorithm
*      - \ref DistributedStep2LocalPlusPlusInputId
*      - \ref DistributedLocalPlusPlusInputDataId Identifiers of input objects for computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*      - \ref DistributedStep2LocalPlusPlusPartialResultId Identifiers of results of computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*
* \par References
*      - DistributedStep2LocalPlusPlusInput class
*      - DistributedStep2LocalPlusPlusPartialResult class
*/
template <typename algorithmFPType, Method method>
class Distributed<step2Local, algorithmFPType, method> : public DistributedStep2LocalPlusPlusBase
{
public:
    typedef algorithms::kmeans::init::DistributedStep2LocalPlusPlusInput InputType;
    typedef algorithms::kmeans::init::DistributedStep2LocalPlusPlusPartialResult PartialResultType;

    /**
    *  Main constructor
    *  \param[in] nClusters        Number of clusters
    *  \param[in] bFirstIteration  true if this is the first iteration in the loop of steps 2-4.
    */
    Distributed(size_t nClusters, bool bFirstIteration);

    /**
    * Copy constructor
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    Distributed(const Distributed<step2Local, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
    * Returns the structure that contains computed partial results
    * \return Structure that contains computed partial results
    */
    DistributedStep2LocalPlusPlusPartialResultPtr getPartialResult() { return _partialResult; }

    /**
    * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
    * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
    */
    services::Status setPartialResult(const DistributedStep2LocalPlusPlusPartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
    * Validates the parameters of the finalizeCompute() method
    */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE { return services::Status(); }

    /**
    * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
    * with a copy of input objects and parameters of this algorithm
    * \return Pointer to the newly allocated algorithm
    */
    services::SharedPtr<Distributed<step2Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step2Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new DistributedStep2LocalPlusPlusPartialResult());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->initialize(&input, _par, (int)method);
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Local, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< Step2 parameters structure */

private:
    DistributedStep2LocalPlusPlusPartialResultPtr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP3MASTER_ALGORITHMFPTYPE_METHOD"></a>
* \brief Computes initial clusters for K-Means algorithm in the 3rd step of the distributed processing mode.
*        Used with plusPlus and parallelPlus methods only on the master node.
* <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
* \tparam method            Method of computing initial clusters for the algorithm, \ref Method
*
* \par Enumerations
*      - \ref Method   Methods of computing initial clusters for K-Means algorithm
*      - \ref DistributedStep3MasterPlusPlusInputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*      - \ref DistributedStep3MasterPlusPlusPartialResultId Identifiers of results of computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*
* \par References
*      - DistributedStep3MasterPlusPlusInput class
*      - DistributedStep3MasterPlusPlusPartialResult class
*/
template <typename algorithmFPType, Method method>
class Distributed<step3Master, algorithmFPType, method> : public DistributedBase
{
public:
    typedef algorithms::kmeans::init::DistributedStep3MasterPlusPlusInput InputType;
    typedef algorithms::kmeans::init::DistributedStep3MasterPlusPlusPartialResult PartialResultType;

    /**
    *  Main constructor
    *  \param[in] nClusters   Number of clusters
    */
    Distributed(size_t nClusters);
    /**
    * Copy constructor
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    Distributed(const Distributed<step3Master, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
    * Returns the structure that contains computed partial results
    * \return Structure that contains computed partial results
    */
    DistributedStep3MasterPlusPlusPartialResultPtr getPartialResult() { return _partialResult; }

    /**
    * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
    * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
    */
    services::Status setPartialResult(const DistributedStep3MasterPlusPlusPartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
    * Validates the parameters of the finalizeCompute() method
    */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE { return services::Status(); }

    /**
    * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
    * with a copy of input objects and parameters of this algorithm
    * \return Pointer to the newly allocated algorithm
    */
    services::SharedPtr<Distributed<step3Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step3Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult->initialize(&input, _par, (int)method);
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Master, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< K-means init parameters structure */

private:
    DistributedStep3MasterPlusPlusPartialResultPtr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP4LOCAL_ALGORITHMFPTYPE_METHOD"></a>
* \brief Computes initial clusters for K-Means algorithm in the 4th step of the distributed processing mode.
*        Used with plusPlus and parallelPlus methods only on a local node.
* <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
* \tparam method            Method of computing initial clusters for the algorithm, \ref Method
*
* \par Enumerations
*      - \ref Method   Methods of computing initial clusters for K-Means algorithm
*      - \ref DistributedStep4LocalPlusPlusInputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*      - \ref DistributedStep4LocalPlusPlusPartialResultId Identifiers of results of computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*
* \par References
*      - DistributedStep4LocalPlusPlusInput class
*      - DistributedStep4LocalPlusPlusPartialResult class
*/
template <typename algorithmFPType, Method method>
class Distributed<step4Local, algorithmFPType, method> : public DistributedBase
{
public:
    typedef algorithms::kmeans::init::DistributedStep4LocalPlusPlusInput InputType;
    typedef algorithms::kmeans::init::DistributedStep4LocalPlusPlusPartialResult PartialResultType;

    /**
    *  Main constructor
    *  \param[in] nClusters   Number of clusters
    */
    Distributed(size_t nClusters);
    /**
    * Copy constructor
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    Distributed(const Distributed<step4Local, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
    * Returns the structure that contains computed partial results
    * \return Structure that contains computed partial results
    */
    DistributedStep4LocalPlusPlusPartialResultPtr getPartialResult() { return _partialResult; }

    /**
    * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
    * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
    */
    services::Status setPartialResult(const DistributedStep4LocalPlusPlusPartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
    * Validates the parameters of the finalizeCompute() method
    */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE { return services::Status(); }

    /**
    * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
    * with a copy of input objects and parameters of this algorithm
    * \return Pointer to the newly allocated algorithm
    */
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< K-means init parameters structure */

private:
    DistributedStep4LocalPlusPlusPartialResultPtr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTED_STEP5MASTER_ALGORITHMFPTYPE_METHOD"></a>
* \brief Computes initial clusters for K-Means algorithm in the 5th step of the distributed processing mode.
*        Used with parallelPlus method only.
* <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
*
* \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
* \tparam method            Method of computing initial clusters for the algorithm, \ref Method
*
* \par Enumerations
*      - \ref Method   Methods of computing initial clusters for K-Means algorithm
*      - \ref DistributedStep5MasterPlusPlusInputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*      - \ref DistributedStep5MasterPlusPlusPartialResultId Identifiers of results of computing initial clusters for K-Means algorithm
*             used with plusPlus and parallelPlus methods only.
*
* \par References
*      - DistributedStep5MasterPlusPlusInput class
*      - DistributedStep5MasterPlusPlusPartialResult class
*/
template <typename algorithmFPType, Method method>
class Distributed<step5Master, algorithmFPType, method> : public DistributedBase
{
public:
    typedef algorithms::kmeans::init::DistributedStep5MasterPlusPlusInput InputType;
    typedef algorithms::kmeans::init::Result ResultType;
    typedef algorithms::kmeans::init::DistributedStep5MasterPlusPlusPartialResult PartialResultType;

    /**
    *  Main constructor
    *  \param[in] nClusters   Number of clusters
    */
    Distributed(size_t nClusters);

    /**
    * Copy constructor
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    Distributed(const Distributed<step5Master, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
    * Returns the structure that contains the results of computing initial clusters for K-Means algorithm
    * \return Structure that contains the results of computing initial clusters for K-Means algorithm
    */
    ResultPtr getResult() { return _result; }

    /**
    * Registers user-allocated memory to store the results of computing initial clusters for K-Means algorithm
    * \param[in] result  Structure to store the results of computing initial clusters for K-Means algorithm
    */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
    * Returns the structure that contains computed partial results
    * \return Structure that contains computed partial results
    */
    DistributedStep5MasterPlusPlusPartialResultPtr getPartialResult() { return _partialResult; }

    /**
    * Registers user-allocated memory to store partial results of computing initial clusters for K-Means algorithm
    * \param[in] partialRes  Structure to store partial results of computing initial clusters for K-Means algorithm
    */
    services::Status setPartialResult(const DistributedStep5MasterPlusPlusPartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
    * Validates the parameters of the finalizeCompute() method
    */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE { return services::Status(); }

    /**
    * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
    * with a copy of input objects and parameters of this algorithm
    * \return Pointer to the newly allocated algorithm
    */
    services::SharedPtr<Distributed<step5Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step5Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step5Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step5Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step5Master, algorithmFPType, method)(&_env);
        _in                        = &input;
    }

public:
    InputType input;           /*!< %Input data structure */
    ParameterType & parameter; /*!< Parameters structure */

private:
    DistributedStep5MasterPlusPlusPartialResultPtr _partialResult;
    ResultPtr _result;

    Distributed & operator=(const Distributed &);
};
} // namespace interface2
using interface2::DistributedContainer;
using interface2::DistributedBase;
using interface2::DistributedStep2LocalPlusPlusBase;
using interface2::Distributed;
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
#endif
