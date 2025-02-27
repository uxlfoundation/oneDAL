/* file: pca_distributed.h */
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
//  Implementation of the interface for the PCA algorithm in the distributed
// processing mode
//--
*/

#ifndef __PCA_DISTRIBUTED_H__
#define __PCA_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/pca/pca_types.h"
#include "algorithms/pca/pca_online.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
/**
 * @defgroup pca_distributed Distributed
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm in the distributed processing mode
 */
template <ComputeStep computeStep, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm on the local node
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, correlationDense, cpu> : public OnlineContainer<algorithmFPType, correlationDense, cpu>
{
public:
    /** \brief Constructor */
    DistributedContainer(daal::services::Environment::env * daalEnv) : OnlineContainer<algorithmFPType, correlationDense, cpu>(daalEnv) {};
    virtual ~DistributedContainer() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm on the master node
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();

    /**
     * Computes a partial result of the PCA algorithm in the second step
     * of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the second step
     * of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm on the local node
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, svdDense, cpu> : public OnlineContainer<algorithmFPType, svdDense, cpu>
{
public:
    /** \brief Constructor */
    DistributedContainer(daal::services::Environment::env * daalEnv) : OnlineContainer<algorithmFPType, svdDense, cpu>(daalEnv) {};
    virtual ~DistributedContainer() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm on the master node
 */
template <typename algorithmFPType, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of the PCA algorithm in the second step
     * of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes thel result of the PCA algorithm in the second step
     * of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED"></a>
 * \brief Computes the result of the PCA algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the PCA algorithm, double or float
 * \tparam method           Computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the algorithm
 *
 * \par References
 *      - \ref interface1::DistributedParameter class
 */
template <ComputeStep computeStep, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = correlationDense>
class DAAL_EXPORT Distributed : public Analysis<distributed>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the PCA algorithm on the local nodes
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 * \tparam method           Computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the algorithm
 *
 * \par References
 *      - \ref interface1::DistributedParameter class
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

    /** Default constructor */
    Distributed() : Online<algorithmFPType, method>() {}

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other) : Online<algorithmFPType, method>(other) {}

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Computes the result of the PCA Correlation algorithm on local nodes
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template <typename algorithmFPType>
class Distributed<step2Master, algorithmFPType, correlationDense> : public Analysis<distributed>
{
public:
    typedef algorithms::pca::DistributedInput<correlationDense> InputType;
    typedef algorithms::pca::DistributedParameter<step2Master, algorithmFPType, correlationDense> ParameterType;
    typedef algorithms::pca::Result ResultType;
    typedef algorithms::pca::PartialResult<correlationDense> PartialResultType;

    /** Default constructor */
    Distributed();

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, correlationDense> & other);

    ~Distributed() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)correlationDense; }

    /**
     * Registers user-allocated memory to store  partial results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial results of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<correlationDense> > & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns structure that contains computed partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<correlationDense> > getPartialResult()
    {
        _partialResult->set(sumCorrelation, parameter.covariance->getPartialResult()->get(covariance::sum));
        _partialResult->set(nObservationsCorrelation, parameter.covariance->getPartialResult()->get(covariance::nObservations));
        _partialResult->set(crossProductCorrelation, parameter.covariance->getPartialResult()->get(covariance::crossProduct));
        return _partialResult;
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, correlationDense> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, correlationDense> >(cloneImpl());
    }

    DistributedInput<correlationDense> input;                                       /*!< Input object */
    DistributedParameter<step2Master, algorithmFPType, correlationDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<correlationDense> > _partialResult;
    ResultPtr _result;

    virtual Distributed<step2Master, algorithmFPType, correlationDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, correlationDense>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, correlationDense);
        _res               = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, correlationDense);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, correlationDense)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<correlationDense>());
        _result.reset(new ResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_SVDDENSE"></a>
 * \brief Computes the result of the PCA SVD algorithm on local nodes
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template <typename algorithmFPType>
class Distributed<step2Master, algorithmFPType, svdDense> : public Analysis<distributed>
{
public:
    typedef algorithms::pca::DistributedInput<svdDense> InputType;
    typedef algorithms::pca::DistributedParameter<step2Master, algorithmFPType, svdDense> ParameterType;
    typedef algorithms::pca::Result ResultType;
    typedef algorithms::pca::PartialResult<svdDense> PartialResultType;

    /** Default constructor */
    Distributed();

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, svdDense> & other);

    ~Distributed() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)svdDense; }

    /**
     * Registers user-allocated memory to store  partial results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial results of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<svdDense> > & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns structure that contains computed partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<svdDense> > getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, svdDense> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, svdDense> >(cloneImpl());
    }

    DistributedInput<svdDense> input;                                       /*!< Input object */
    DistributedParameter<step2Master, algorithmFPType, svdDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<svdDense> > _partialResult;
    ResultPtr _result;

    virtual Distributed<step2Master, algorithmFPType, svdDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, svdDense>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, svdDense);
        _res               = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, svdDense);
        _pres              = _partialResult.get();
        return s;
    }
    services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, svdDense)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<svdDense>());
        _result.reset(new ResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
