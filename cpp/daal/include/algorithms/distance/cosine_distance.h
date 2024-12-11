/* file: cosine_distance.h */
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
//  Implementation of the interface for the cosine distance algorithm in the batch
//  processing mode
//--
*/

#ifndef __COSDISTANCE_H__
#define __COSDISTANCE_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/distance/cosine_distance_types.h"

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace interface1
{
/**
 * @defgroup cosine_distance_batch Batch
 * @ingroup cosine_distance
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSINE_DISTANCE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the cosine distance algorithm.
 *        This class is associated with daal::algorithms::cosine_distance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the cosine distance, double or float
 * \tparam method           Cosine distance computation method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the cosine distance algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default constructor */
    ~BatchContainer();
    /**  Delete copy-constructor and copy-assignment constructor to follow the rule of three */
    BatchContainer(const BatchContainer &)             = delete;
    BatchContainer & operator=(const BatchContainer &) = delete;

    /**
     * Computes the result of the cosine distance algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSINE_DISTANCE__BATCH"></a>
 * \brief Computes the cosine distance in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-COSDISTANCE-ALGORITHM">Cosine distance algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the cosine distance, double or float
 * \tparam method           Cosine distance computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Cosine distance computation methods
 *      - \ref InputId  Identifiers of cosine distance input objects
 *      - \ref ResultId Identifiers of cosine distance results
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::cosine_distance::Input InputType;
    typedef algorithms::cosine_distance::Result ResultType;

    Batch() { initialize(); }

    /**
     * Constructs a cosine distance algorithm by copying input objects
     * of another cosine distance algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other) : input(other.input) { initialize(); }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the cosine distance
     * \return Structure that contains the cosine distance
     */
    ResultPtr getResult() { return _result; }

    /**
     * Sets the memory to store results of the cosine distance algorithm
     * \param[in] res  Structure to store results of the algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated cosine distance algorithm with a copy of input objects
     * of this cosine distance algorithm
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
    InputType input; /*!< %Input objects of the algorithm */

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace cosine_distance
} // namespace algorithms
} // namespace daal
#endif
