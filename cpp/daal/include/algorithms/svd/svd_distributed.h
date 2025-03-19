/* file: svd_distributed.h */
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
//  Implementation of the interface of the SVD algorithm in the distributed processing mode
//--
*/

#ifndef __SVD_DISTRIBUTED_H__
#define __SVD_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/svd/svd_types.h"
#include "algorithms/svd/svd_online.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{
/**
 * @defgroup svd_distributed Distributed
 * @ingroup svd
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm.
 *
 * \tparam step             Step of the computing algorithm in the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::svd::Method
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the first step of the SVD algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public OnlineContainer<algorithmFPType, method, cpu>
{
public:
    /** Default constructor */
    DistributedContainer(daal::services::Environment::env * daalEnv) : OnlineContainer<algorithmFPType, method, cpu>(daalEnv) {}
    /** Default destructor */
    virtual ~DistributedContainer() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the second step of the SVD algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the SVD algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the SVD algorithm in the second step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the third step of the SVD algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the SVD algorithm in the third step
     * of the distributed processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the SVD algorithm in the third step
     * of the distributed processing mode
     */
    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTED"></a>
 * \brief Computes results of the SVD algorithm in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam step             One of the three possible steps of the SVD algorithm in the distributed processing mode
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   SVD computation methods
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed : public daal::algorithms::Analysis<distributed>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Runs the first step of the SVD algorithm in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    typedef Online<algorithmFPType, method> super;

    typedef typename super::InputType InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;
    typedef typename super::PartialResultType PartialResultType;

    Distributed() : Online<algorithmFPType, method>() {}

    /**
     * Constructs an SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other) : Online<algorithmFPType, method>(other) {}

    /**
     * Returns a pointer to the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
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

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Runs the second step of the SVD algorithm in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   SVD computation methods
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::svd::DistributedStep2Input InputType;
    typedef algorithms::svd::Parameter ParameterType;
    typedef algorithms::svd::Result ResultType;
    typedef algorithms::svd::DistributedPartialResult PartialResultType;

    InputType input;         /*!< %DistributedStep2Input data structure */
    ParameterType parameter; /*!< SVD parameters structure */

    Distributed();

    /**
     * Constructs a QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> & other);
    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    ResultPtr getResult() { return _partialResult->get(finalResultFromStep2Master); }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    DistributedPartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store computed results of the SVD algorithm
     */
    services::Status setPartialResult(const DistributedPartialResultPtr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        DAAL_CHECK(partialRes->get(finalResultFromStep2Master), services::ErrorNullResult)
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Validates parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        if (!_partialResult) return services::Status(services::ErrorNullResult);
        return _partialResult->check(_par, method);
    }

    /**
     * Returns a pointer to the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
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

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        _pres = _partialResult.get();
        return services::Status();
    }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in                        = &input;
        _par                       = &parameter;
    }

private:
    DistributedPartialResultPtr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Runs the third step of the SVD algorithm in the distributed processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   SVD computation methods
 */
template <typename algorithmFPType, Method method>
class Distributed<step3Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::svd::DistributedStep3Input InputType;
    typedef algorithms::svd::Parameter ParameterType;
    typedef algorithms::svd::Result ResultType;
    typedef algorithms::svd::DistributedPartialResultStep3 PartialResultType;

    InputType input;         /*!< %DistributedStep3Input data structure */
    ParameterType parameter; /*!< SVD parameters structure */

    Distributed();

    /**
     * Constructs an SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> & other);

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    ResultPtr getResult() { return _partialResult->get(finalResultFromStep3); }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    DistributedPartialResultStep3Ptr getPartialResult() { return _partialResult; }

    /**
    * Registers user-allocated memory to store computed results of the SVD algorithm

    */
    services::Status setPartialResult(const DistributedPartialResultStep3Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        DAAL_CHECK(partialRes->get(finalResultFromStep3), services::ErrorNullResult)
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
    * Registers user-allocated memory to store computed results of the SVD algorithm

    */
    services::Status setResult(const ResultPtr & res) { return services::Status(); }

    /**
     * Validates parameters of the finalizeCompute() method
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        if (!_partialResult) return services::Status(services::ErrorNullResult);
        return _partialResult->check(_par, method);
    }

    /**
     * Returns a pointer to the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());

        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        if (!s) return s;

        data_management::DataCollectionPtr qCollection = input.get(inputOfStep3FromStep1);

        s = _partialResult->setPartialResultStorage<algorithmFPType>(qCollection.get());

        _pres = _partialResult.get();
        return s;
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _par                       = &parameter;
    }

private:
    DistributedPartialResultStep3Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace svd
} // namespace algorithms
} // namespace daal
#endif
