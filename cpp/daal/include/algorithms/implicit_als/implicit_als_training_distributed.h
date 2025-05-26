/* file: implicit_als_training_distributed.h */
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
//  Implementation of the interface for implicit ALS model-based training in the
//  distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_DISTRIBUTED_H__
#define __IMPLICIT_ALS_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_training_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace interface1
{
/**
 * @defgroup implicit_als_training_distributed Distributed
 * @ingroup implicit_als_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the result of implicit ALS model-based training
 * in the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the second step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the second step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the third step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the third step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the fourth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step4Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the fourth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based training
     * in the fourth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based training
     * in the fourth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED"></a>
 * \brief Trains the implicit ALS model in the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm
 *                          in the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method     %Training methods of the implicit ALS algorithm in the first step of the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = fastCSR>
class DAAL_EXPORT Distributed : public Training<distributed>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the first step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the first step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  %Training methods of the implicit ALS algorithm in the first step of the distributed processing mode
 *
 * \par References
 *      - \ref DistributedInput<step1Local> class
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::DistributedInput<step1Local> InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::training::Result ResultType;
    typedef algorithms::implicit_als::training::DistributedPartialResultStep1 PartialResultType;

    DistributedInput<step1Local> input; /*!< %Input data structure */
    ParameterType parameter;            /*!< %Training \ref implicit_als::interface1::Parameter "parameters" */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
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
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     */
    services::Status setPartialResult(const DistributedPartialResultStep1Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     */
    DistributedPartialResultStep1Ptr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    DistributedPartialResultStep1Ptr _partialResult;

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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the second step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the second step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  %Training methods of the implicit ALS algorithm in the second step of the distributed processing mode
 *
 * \par References
 *      - \ref DistributedInput<step2Master> class
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Master, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::DistributedInput<step2Master> InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::training::Result ResultType;
    typedef algorithms::implicit_als::training::DistributedPartialResultStep2 PartialResultType;

    DistributedInput<step2Master> input; /*!< %Input data structure */
    ParameterType parameter;             /*!< %Training \ref implicit_als::interface1::Parameter "parameters" */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm\
     * in the second step of the distributed processing mode
     */
    services::Status setPartialResult(const DistributedPartialResultStep2Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the second step of the distributed processing mode
     */
    DistributedPartialResultStep2Ptr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    DistributedPartialResultStep2Ptr _partialResult;

    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the third step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the third step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, Method
 *
 * \par Enumerations
 *      - \ref Method  %Training methods of the implicit ALS algorithm in the third step of the distributed processing mode
 *
 * \par References
 *      - \ref DistributedInput<step3Local> class
 */
template <typename algorithmFPType, Method method>
class Distributed<step3Local, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::DistributedInput<step3Local> InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::training::Result ResultType;
    typedef algorithms::implicit_als::training::DistributedPartialResultStep3 PartialResultType;

    DistributedInput<step3Local> input; /*!< %Input data structure */
    ParameterType parameter;            /*!< %Training \ref implicit_als::interface1::Parameter "parameters" */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> & other);

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     */
    services::Status setPartialResult(const DistributedPartialResultStep3Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the third step of the distributed processing mode
     */
    DistributedPartialResultStep3Ptr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    DistributedPartialResultStep3Ptr _partialResult;

    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTED_STEP4LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Trains the implicit ALS model in the fourth step of the distributed processing mode
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the implicit ALS training algorithm in the fourth step
 *                          of the distributed processing mode, double or float
 * \tparam method           Implicit ALS training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  %Training methods of the implicit ALS algorithm in the fourth step of the distributed processing mode
 *
 * \par References
 *      - \ref DistributedInput<step4Local> class
 */
template <typename algorithmFPType, Method method>
class Distributed<step4Local, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::implicit_als::training::DistributedInput<step4Local> InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::training::Result ResultType;
    typedef algorithms::implicit_als::training::DistributedPartialResultStep4 PartialResultType;

    DistributedInput<step4Local> input; /*!< %Input data structure */
    ParameterType parameter;            /*!< %Training \ref implicit_als::interface1::Parameter "parameters" */

    /** Default constructor */
    Distributed();

    /**
     * Constructs an implicit ALS training algorithm by copying input objects and parameters
     * of another implicit ALS training algorithm
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
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * \param[in] partialResult  Structure to store partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     */
    services::Status setPartialResult(const DistributedPartialResultStep4Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * \return Structure that contains partial results of the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     */
    DistributedPartialResultStep4Ptr getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to the newly allocated ALS training algorithm with a copy of input objects
     * and parameters of this ALS training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    DistributedPartialResultStep4Ptr _partialResult;

    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
