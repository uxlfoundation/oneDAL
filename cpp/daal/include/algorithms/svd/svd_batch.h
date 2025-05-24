/* file: svd_batch.h */
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
//  Implementation of the interface for the SVD algorithm in the batch processing mode
//--
*/

#ifndef __SVD_BATCH_H__
#define __SVD_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/svd/svd_types.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{
/**
 * @defgroup svd_batch Batch
 * @ingroup svd
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the SVD algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__BATCH"></a>
 * \brief Computes results of the SVD algorithm in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::svd::Input InputType;
    typedef algorithms::svd::Parameter ParameterType;
    typedef algorithms::svd::Result ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< SVD parameters structure */

    Batch();

    /**
     * Constructs an SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed results of the SVD algorithm
     * \return Structure that contains computed results of the SVD algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store computed results of the SVD algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_in, 0, 0);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
    }

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace svd
} // namespace algorithms
} // namespace daal
#endif
