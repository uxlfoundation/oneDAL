/* file: low_order_moments_batch.h */
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
//  Implementation of the interface for the low order moments algorithm in the
//  batch processing mode
//--
*/

#ifndef __LOW_ORDER_MOMENTS_BATCH_H__
#define __LOW_ORDER_MOMENTS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/moments/low_order_moments_types.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace interface1
{
/**
 * @defgroup low_order_moments_batch Batch
 * @ingroup low_order_moments
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCHCONTAINERIFACE"></a>
* \brief Class that specifies interfaces of implementations of the low order moments algorithm.
*/
class BatchContainerIface : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    BatchContainerIface() {}
    virtual ~BatchContainerIface() {}

    /**
     * Computes the result of the low order moments algorithm in the batch processing mode
     */
    virtual services::Status compute() = 0;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the low order moments algorithm.
 *        This class is associated with daal::algorithms::low_order_moments::Batch class

 *
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::low_order_moments::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the low order moments algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCHIFACE"></a>
 * \brief Abstract class that specifies interface of the algorithms
 *        for computing moments of low order in the batch processing mode
 */
class DAAL_EXPORT BatchImpl : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::low_order_moments::Input InputType;
    typedef algorithms::low_order_moments::Parameter ParameterType;
    typedef algorithms::low_order_moments::Result ResultType;

    /** Default constructor */
    BatchImpl() { initialize(); }

    /**
     * Constructs an algorithm for moments of low order computation
     * by copying input objects and parameters of another algorithm for moments of low order computation
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    BatchImpl(const BatchImpl & other) : input(other.input), parameter(other.parameter) { initialize(); }

    /**
     * Returns the structure that contains moments of low order
     * \return Structure that contains the computed matrix
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of computation of moments of low order
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
     * Returns a pointer to the newly allocated algorithm for moments of low order computation
     * with a copy of input objects and parameters of this algorithm for moments of low order
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<BatchImpl> clone() const { return services::SharedPtr<BatchImpl>(cloneImpl()); }

    virtual ~BatchImpl() {}

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Algorithm parameter */

protected:
    ResultPtr _result;

    void initialize()
    {
        _result.reset(new ResultType());
        _in  = &input;
        _par = &parameter;
    }
    virtual BatchImpl * cloneImpl() const DAAL_C11_OVERRIDE = 0;

private:
    BatchImpl & operator=(const BatchImpl &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCH"></a>
 * \brief Computes moments of low order in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order moments algorithm description and usage models</a> -->
 *
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::low_order_moments::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for moments of low order
 *      - \ref InputId  Identifiers of input objetcs for the low order moments algorithm
 *      - \ref ResultId Identifiers of results of the low order moments algorithm
 *
 * \par References
 *      - Input class
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
     * Constructs an algorithm that computes moments of low order by copying input objects
     * of another algorithm that computes moments of low order
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
     * Returns a pointer to the newly allocated algorithm that computes moments of low order
     * with a copy of input objects of this algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, 0, 0);
        _res               = _result.get();
        return s;
    }

    void initialize() { this->_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env); }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainerIface;
using interface1::BatchContainer;
using interface1::BatchImpl;
using interface1::Batch;

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
#endif
