/* file: minmax.h */
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
//  Implementation of the interface for the min-max normalization algorithm
//  in the batch processing mode
//--
*/

#ifndef __MINMAX_BATCH_H__
#define __MINMAX_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/normalization/minmax_types.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{
/** @defgroup minmax_batch Batch
 * @ingroup minmax
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__MINMAX__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the min-max normalization algorithm.
 *        It is associated with the daal::algorithms::normalization::minmax::Batch class
 *        and supports methods of min-max normalization computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the min-max normalization algorithms, double or float
 * \tparam method           Min-max normalization computation method, daal::algorithms::normalization::minmax::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the min-max normalization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    virtual ~BatchContainer();

    /**
     * Computes the result of the min-max normalization algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__MINMAX__BATCH"></a>
 * \brief Normalizes datasets in the batch processing mode
 * <!-- \n<a href="DAAL-REF-MINMAX-ALGORITHM">Min-max normalization algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the min-max normalization, double or float
 * \tparam method           Min-max normalization computation method, daal::algorithms::normalization::minmax::Method
 *
 * \par Enumerations
 *      - daal::algorithms::normalization::minmax::Method   Min-max normalization computation methods
 *      - daal::algorithms::normalization::minmax::InputId  Identifiers of min-max normalization input objects
 *      - daal::algorithms::normalization::minmax::ResultId Identifiers of min-max normalization results
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::normalization::minmax::Input InputType;
    typedef algorithms::normalization::minmax::Parameter<algorithmFPType> ParameterType;
    typedef algorithms::normalization::minmax::Result ResultType;

    InputType input;                      /*!< %input data structure */
    Parameter<algorithmFPType> parameter; /*!< Parameters */

    /** Default constructor */
    Batch();

    /**
     * Constructs min-max normalization algorithm by copying input objects
     * of another min-max normalization algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed results of the min-max normalization
     * \return Structure that contains computed results of the min-max normalization
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of the min-max normalization algorithms
     * \param[in] result Structure to store results of the min-max normalization algorithms
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
     * Returns a pointer to the newly allocated min-max normalization algorithm
     * with a copy of input objects of this min-max normalization algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, method);
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

} // namespace minmax
} // namespace normalization
} // namespace algorithms
} // namespace daal
#endif
