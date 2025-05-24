/* file: implicit_als_training_init_distributed.h */
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
//  Implementation of the interface for the implicit ALS initialization algorithm
//  in the distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_H__
#define __IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_training_init_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace interface1
{
/**
 * @defgroup implicit_als_init_distributed Distributed
 * @ingroup implicit_als_init
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the results of the implicit ALS initialization algorithm
 * in the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();
    /**
     * Computes a partial result of the implicit ALS initialization algorithm
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the implicit ALS initialization algorithm
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the third step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();
    /**
     * Computes a partial result of the implicit ALS initialization algorithm
     * in the third step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the implicit ALS initialization algorithm
     * in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTED"></a>
 * \brief Initializes the implicit ALS model in the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS initialization algorithm
 *                          in the distributed processing mode, double or float
 * \tparam method           Implicit ALS initialization method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Initialization methods of the implicit ALS algorithm in the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = fastCSR>
class DAAL_EXPORT Distributed
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Initializes the implicit ALS model in the first step of the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS initialization algorithm
 *                          in the distributed processing mode, double or float
 * \tparam method           Implicit ALS initialization method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Initialization methods of the implicit ALS algorithm in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::init::DistributedInput<step1Local> InputType;
    typedef algorithms::implicit_als::training::init::DistributedParameter ParameterType;
    typedef algorithms::implicit_als::training::init::Result ResultType;
    typedef algorithms::implicit_als::training::init::PartialResult PartialResultType;

    DistributedInput<step1Local> input; /*!< %Input data structure */
    ParameterType parameter;            /*!< Parameters of the implicit ALS initialization algorithm */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an algorithm for initializing the implicit ALS model by copying input objects and parameters
     * of another algorithm for initializing the implicit ALS model
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
     * Registers user-allocated memory to store partial results of the implicit ALS initialization algorithm
     * \param[in] partialResult  Structure to store partial results of the implicit ALS initialization algorithm
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS initialization algorithm
     * \return Structure that contains partial results of the implicit ALS initialization algorithm
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated algorithm for initializing the implicit ALS model
     * with a copy of input objects and parameters of this algorithm for initializing the implicit ALS model
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    PartialResultPtr _partialResult;

    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, method);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTED_STEP2LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Initializes the implicit ALS model in the second step of the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS initialization algorithm
 *                          in the distributed processing mode, double or float
 * \tparam method           Implicit ALS initialization method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Initialization methods of the implicit ALS algorithm in the second step of the distributed processing mode
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Local, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::init::DistributedInput<step2Local> InputType;
    typedef algorithms::implicit_als::training::init::Result ResultType;
    typedef algorithms::implicit_als::training::init::DistributedPartialResultStep2 PartialResultType;

    DistributedInput<step2Local> input; /*!< %Input data structure */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an algorithm for initializing the implicit ALS model by copying input objects
     * of another algorithm for initializing the implicit ALS model
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  of the algorithm
     */
    Distributed(const Distributed<step2Local, algorithmFPType, method> & other);

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS initialization algorithm
     * \param[in] partialResult  Structure to store partial results of the implicit ALS initialization algorithm
     */
    services::Status setPartialResult(const DistributedPartialResultStep2Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS initialization algorithm
     * \return Structure that contains partial results of the implicit ALS initialization algorithm
     */
    DistributedPartialResultStep2Ptr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated algorithm for initializing the implicit ALS model
     * with a copy of input objects and parameters of this algorithm for initializing the implicit ALS model
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    DistributedPartialResultStep2Ptr _partialResult;

    virtual Distributed<step2Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, NULL, method);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = NULL;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
