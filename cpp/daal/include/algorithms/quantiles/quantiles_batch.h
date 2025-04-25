/* file: quantiles_batch.h */
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
//  Implementation of the interface for the quantiles algorithm in the batch processing mode
//--
*/

#ifndef __QUANTILES_BATCH_H__
#define __QUANTILES_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/quantiles/quantiles_types.h"

namespace daal
{
namespace algorithms
{
namespace quantiles
{
namespace interface1
{
/**
 * @defgroup quantiles_batch Batch
 * @ingroup quantiles
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the quantiles algorithm.
 *        It is associated with the daal::algorithms::quantiles::Batch class
 *        and supports methods of quantiles computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the quantile algorithms, double or float
 * \tparam method           Quantiles computation method, \ref daal::algorithms::quantiles::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the quantiles algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the quantiles algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__BATCH"></a>
 * \brief Computes values of quantiles in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-QUANTILES-ALGORITHM">Quantiles algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the quantile algorithms, double or float
 * \tparam method           Quantiles computation method, \ref daal::algorithms::quantiles::Method
 *
 * \par Enumerations
 *      - \ref Method   Quantiles computation methods
 *      - \ref InputId  Identifiers of quantiles input objects
 *      - \ref ResultId Identifiers of quantiles results
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::quantiles::Input InputType;
    typedef algorithms::quantiles::Parameter ParameterType;
    typedef algorithms::quantiles::Result ResultType;

    InputType input;         /*!< %input data structure */
    ParameterType parameter; /*!< Quantiles parameters structure */

    /** Default constructor     */
    Batch();

    /**
     * Constructs algorithm that computes quantiles by copying input objects and parameters
     * of another algorithm
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
     * Returns the structure that contains computed results of the quantile algorithms
     * \return Structure that contains computed results of the quantile algorithms
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of the quantile algorithms
     * \param[in] result Structure to store results of the quantile algorithms
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        if (!result) return services::Status(services::ErrorNullResult);
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated algorithm that computes quantiles
     * with a copy of input objects and parameters of this algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
        _result.reset(new ResultType());
    }

    ResultPtr _result;

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace quantiles
} // namespace algorithms
} // namespace daal
#endif
