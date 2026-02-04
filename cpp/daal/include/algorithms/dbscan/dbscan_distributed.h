/* file: dbscan_distributed.h */
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
//  Implementation of the interface for the DBSCAN algorithm in the distributed
//  processing mode
//--
*/

#ifndef __DBSCAN_DISTRIBUTED_H__
#define __DBSCAN_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/dbscan/dbscan_types.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{
/**
 * @defgroup dbscan_distributed Distributed
 * @ingroup dbscan_compute
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the result of DBSCAN algorithm
 * in the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the first step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the second step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the second step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the third step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the third step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the fourth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step4Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the fourth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the fourth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the fourth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the fifth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step5Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the fifth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the fifth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the fifth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP6LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the sixth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step6Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the sixth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the sixth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the sixth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP7MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the seventh step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step7Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the seventh step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the seventh step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the seventh step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP8LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the eighth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step8Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the eighth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the eighth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the eighth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP9MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the ninth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step9Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the ninth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the ninth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the ninth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP10LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the tenth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step10Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the tenth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the tenth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the tenth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP11LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the eleventh step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step11Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the eleventh step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the eleventh step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the eleventh step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP12LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the twelfth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step12Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the twelfth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the twelfth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the twelfth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDCONTAINER_STEP13LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for the DBSCAN algorithm in the thirteenth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step13Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for DBSCAN algorithm with a specified environment
     * in the thirteenth step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of DBSCAN algorithm
     * in the thirteenth step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of DBSCAN algorithm
     * in the thirteenth step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED"></a>
 * \brief Computes the results of the DBSCAN algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 *
 * \par References
 *      - Input class
 *      - Result class
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step1Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep1 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep1Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep1Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep1Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP2LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the second step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step2Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep2 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep2Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep2Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step2Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step2Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep2Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP3LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the third step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step3Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step3Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep3 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] leftBlocks      Number of blocks that will process observations with value of selected
                                   split feature lesser than selected split value
     *  \param[in] rightBlocks     Number of blocks that will process observations with value of selected
                                   split feature greater than selected split value
     */
    Distributed(size_t leftBlocks, size_t rightBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step3Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep3Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep3Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep3Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP4LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the fourth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step4Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step4Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep4 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] leftBlocks      Number of blocks that will process observations with value of selected
                                   split feature lesser than selected split value
     *  \param[in] rightBlocks     Number of blocks that will process observations with value of selected
                                   split feature greater than selected split value
     */
    Distributed(size_t leftBlocks, size_t rightBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step4Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep4Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep4Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep4Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP5LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the fifth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step5Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step5Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep5 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     *  \param[in] epsilon       Radius of neighborhood
     */
    Distributed(size_t blockIndex, size_t nBlocks, algorithmFPType epsilon);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step5Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep5Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep5Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step5Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step5Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step5Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step5Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step5Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep5Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP6LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the sixth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step6Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step6Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep6 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex       Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks          Number of blocks initially passed for computation on all nodes
     *  \param[in] epsilon          Radius of neighborhood
     *  \param[in] minObservations  Minimal total weight of observations in neighborhood of core observation
     */
    Distributed(size_t blockIndex, size_t nBlocks, algorithmFPType epsilon, size_t minObservations);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step6Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep6Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep6Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step6Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step6Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step6Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step6Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step6Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep6Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP7MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the seventh step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step7Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step7Master> InputType;
    typedef algorithms::dbscan::DistributedPartialResultStep7 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     */
    Distributed();

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step7Master, algorithmFPType, method> & other);

    ~Distributed() {}

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep7Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep7Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step7Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step7Master, algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual services::Status checkPartialResult() override { return services::Status(); }

protected:
    virtual Distributed<step7Master, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step7Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return services::Status();
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step7Master, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep7Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP8LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the eighth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step8Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step8Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep8 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step8Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep8Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep8Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step8Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step8Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step8Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step8Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step8Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep8Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP9MASTER_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the seventh step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step9Master, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step9Master> InputType;
    typedef algorithms::dbscan::DistributedResultStep9 ResultType;
    typedef algorithms::dbscan::DistributedPartialResultStep9 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     */
    Distributed();

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step9Master, algorithmFPType, method> & other);

    ~Distributed() {}

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed results
     * \return Structure that contains computed results
     */
    DistributedResultStep9Ptr getResult() { return _result; }

    /**
     * Sets the structure that contains computed results
     */
    services::Status setResult(const DistributedResultStep9Ptr & result)
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
    DistributedPartialResultStep9Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep9Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step9Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step9Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step9Master, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step9Master, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int)method);
        _res               = _result.get();
        return services::Status();
    }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step9Master, algorithmFPType, method)(&_env);
        _in                        = &input;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedResultStep9Ptr _result;
    DistributedPartialResultStep9Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP10LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the tenth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step10Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step10Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep10 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step10Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep10Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep10Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step10Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step10Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step10Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step10Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step10Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep10Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP11LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the eleventh step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step11Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step11Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep11 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step11Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep11Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep11Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step11Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step11Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step11Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step11Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step11Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep11Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP12LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the eighth step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step12Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step12Local> InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::DistributedPartialResultStep12 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     *  \param[in] blockIndex    Unique identifier of block initially passed for computation on the local node
     *  \param[in] nBlocks       Number of blocks initially passed for computation on all nodes
     */
    Distributed(size_t blockIndex, size_t nBlocks);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step12Local, algorithmFPType, method> & other);

    ~Distributed()
    {
        delete _par;
        _par = 0;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    DistributedPartialResultStep12Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep12Ptr & partialRes)
    {
        DAAL_CHECK(partialRes, services::ErrorNullPartialResult);
        _partialResult = partialRes;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step12Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step12Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step12Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step12Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override { return services::Status(); }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step12Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedPartialResultStep12Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTED_STEP13LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Computes the results of the DBSCAN algorithm in the seventh step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the  DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType, Method method>
class Distributed<step13Local, algorithmFPType, method> : public daal::algorithms::Analysis<distributed>
{
public:
    typedef algorithms::dbscan::DistributedInput<step13Local> InputType;
    typedef algorithms::dbscan::DistributedResultStep13 ResultType;
    typedef algorithms::dbscan::DistributedPartialResultStep13 PartialResultType;

    /**
     * Constructs a DBSCAN algorithm
     */
    Distributed();

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step13Local, algorithmFPType, method> & other);

    ~Distributed() {}

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains computed results
     * \return Structure that contains computed results
     */
    DistributedResultStep13Ptr getResult() { return _result; }

    /**
     * Sets the structure that contains computed results
     */
    services::Status setResult(const DistributedResultStep13Ptr & result)
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
    DistributedPartialResultStep13Ptr getPartialResult() { return _partialResult; }

    /**
     * Sets the structure that contains computed partial results
     */
    services::Status setPartialResult(const DistributedPartialResultStep13Ptr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step13Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step13Local, algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual services::Status checkPartialResult() override { return services::Status(); }

protected:
    virtual Distributed<step13Local, algorithmFPType, method> * cloneImpl() const override
    {
        return new Distributed<step13Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() override
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, _par, (int)method);
        _res               = _result.get();
        return services::Status();
    }

    virtual services::Status allocatePartialResult() override
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, _par, (int)method);
        _pres              = _partialResult.get();
        return services::Status();
    }

    virtual services::Status initializePartialResult() override { return services::Status(); }

    void initialize()
    {
        Analysis<distributed>::_ac = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step13Local, algorithmFPType, method)(&_env);
        _in                        = &input;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    DistributedResultStep13Ptr _result;
    DistributedPartialResultStep13Ptr _partialResult;

    Distributed & operator=(const Distributed &);
};

/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace dbscan
} // namespace algorithms
} // namespace daal
#endif
