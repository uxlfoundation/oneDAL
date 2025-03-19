/* file: zscore.h */
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
//  Implementation of the interface for the z-score normalization algorithm
//  in the batch processing mode
//--
*/

#ifndef __ZSCORE_BATCH_H__
#define __ZSCORE_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/normalization/zscore_types.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface3
{
/** @defgroup zscore_batch Batch
 * @ingroup zscore
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the z-score normalization algorithm.
 *        It is associated with the daal::algorithms::normalization::zscore::Batch class
 *        and supports methods of z-score normalization computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the z-score normalization algorithms, double or float
 * \tparam method           Z-score normalization computation method, daal::algorithms::normalization::zscore::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the z-score normalization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the z-score normalization algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__BATCHIFACE"></a>
* \brief Abstract class that specifies interface of the algorithms
*        for computing correlation or variance-covariance matrix in the batch processing mode
*/
class DAAL_EXPORT BatchImpl : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::normalization::zscore::Input InputType;
    typedef algorithms::normalization::zscore::Result ResultType;

    /** Default constructor */
    BatchImpl() { initialize(); };

    /**
    * Constructs an algorithm for correlation or variance-covariance matrix computation
    * by copying input objects and parameters of another algorithm for correlation or variance-covariance
    * matrix computation
    * \param[in] other An algorithm to be used as the source to initialize the input objects
    *                  and parameters of the algorithm
    */
    BatchImpl(const BatchImpl & other) : input(other.input) { initialize(); }

    /**
    * Returns the structure that contains correlation or variance-covariance matrix
    * \return Structure that contains the computed matrix
    */
    ResultPtr getResult() { return _result; };

    /**
    * Returns the pointer to parameter
    * \return Pointer to parameter
    * \DAAL_DEPRECATED_USE{ BatchImpl::parameter }
    */
    DAAL_DEPRECATED_VIRTUAL virtual BaseParameter * getParameter() = 0;

    /**
    * Returns the pointer to parameter
    * \return Pointer to parameter
    */
    virtual BaseParameter & parameter() = 0;

    /**
    * Returns the pointer to parameter
    * \return Pointer to parameter
    */
    virtual const BaseParameter & parameter() const = 0;

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

    InputType input; /*!< %Input data structure */

protected:
    ResultPtr _result;

    void initialize()
    {
        _result = ResultPtr(new ResultType());
        _in     = &input;
    }
    virtual BatchImpl * cloneImpl() const DAAL_C11_OVERRIDE = 0;

private:
    BatchImpl & operator=(const BatchImpl &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__BATCH"></a>
 * \brief Normalizes datasets in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ZSCORE-ALGORITHM">Z-score normalization algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the z-score normalization, double or float
 * \tparam method           Z-score normalization computation method, daal::algorithms::normalization::zscore::Method
 *
 * \par Enumerations
 *      - daal::algorithms::normalization::zscore::Method   Z-score normalization computation methods
 *      - daal::algorithms::normalization::zscore::InputId  Identifiers of z-score normalization input objects
 *      - daal::algorithms::normalization::zscore::ResultId Identifiers of z-score normalization results
 *      - daal::algorithms::normalization::zscore::ResulToComputetId Identifiers of z-score normalization optional result to compute
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public BatchImpl
{
public:
    typedef BatchImpl super;

    typedef typename super::InputType InputType;
    typedef algorithms::normalization::zscore::Parameter<algorithmFPType, method> ParameterType;
    typedef typename super::ResultType ResultType;

    /** Default constructor */
    Batch();

    /**
     * Constructs z-score normalization algorithm by copying input objects
     * of another z-score normalization algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /** Destructor */
    virtual ~Batch() DAAL_C11_OVERRIDE { delete _par; }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    virtual ParameterType & parameter() DAAL_C11_OVERRIDE { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    virtual const ParameterType & parameter() const DAAL_C11_OVERRIDE { return *static_cast<const ParameterType *>(_par); }

    /**
    * Returns the pointer to parameter
    * \return Pointer to parameter
    * \DAAL_DEPRECATED_USE{ Batch::parameter }
    */
    DAAL_DEPRECATED_VIRTUAL virtual BaseParameter * getParameter() DAAL_C11_OVERRIDE { return &(this->Batch::parameter()); }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated z-score normalization algorithm
     * with a copy of input objects of this z-score normalization algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &(this->Batch::parameter()), method);
        _res               = _result.get();
        return s;
    }

    void initialize() { Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env); }

private:
    Batch & operator=(const Batch &);
};

/** @} */
} // namespace interface3
using interface3::BatchContainer;
using interface3::BatchImpl;
using interface3::Batch;

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
#endif
