/* file: cholesky.h */
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
//  Implementation of the interface for the Cholesky decomposition algorithm
//--
*/

#ifndef __CHOLESKY_H__
#define __CHOLESKY_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/cholesky/cholesky_types.h"

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace interface1
{
/**
 * @defgroup cholesky_batch Batch
 * @ingroup cholesky
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Cholesky decomposition algorithm.
 *        This class is associated with daal::algorithms::cholesky::Batch class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Cholesky decomposition algorithm, double or float
 * \tparam method           Cholesky decomposition computation method, \ref daal::algorithms::cholesky::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the Cholesky decomposition algorithm with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the Cholesky decomposition algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__BATCH"></a>
 * \brief Computes Cholesky decomposition in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-CHOLESKY-ALGORITHM">Cholesky decomposition algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Cholesky decomposition algorithm,
 *                          double or float
 * \tparam method           Cholesky decomposition computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for Cholesky decomposition
 *      - \ref InputId  Identifiers of input objects for Cholesky decomposition
 *      - \ref ResultId Result identifiers for the Cholesky decomposition
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::cholesky::Input InputType;
    typedef algorithms::cholesky::Result ResultType;

    /** Default constructor */
    Batch();

    /**
     * Constructs a Cholesky decomposition algorithm by copying input objects
     * of another Cholesky decomposition algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /** Destructor */
    virtual ~Batch() {}

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of Cholesky decomposition
     * \return Structure that contains results of Cholesky decomposition
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of Cholesky decomposition
     * \param[in] result  Structure to store  results of Cholesky decomposition
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Cholesky decomposition algorithm with a copy of input objects
     * of this Cholesky decomposition algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, NULL, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _result.reset(new ResultType());
    }

public:
    InputType input; /*!< %Input data structure */

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace cholesky
} // namespace algorithms
} // namespace daal
#endif
