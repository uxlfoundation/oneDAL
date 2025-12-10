/* file: dbscan_batch.h */
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
//  Implementation of the interface for the DBSCAN algorithm in the batch
//  processing mode
//--
*/

#ifndef __DBSCAN_BATCH_H__
#define __DBSCAN_BATCH_H__

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
 * @defgroup dbscan_batch Batch
 * @ingroup dbscan_compute
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the DBSCAN algorithm.
 *        This class is associated with the daal::algorithms::dbscan::Batch class
 *        and supports the method of DBSCAN computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::dbscan::Method
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the DBSCAN algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the DBSCAN algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__BATCH"></a>
 * \brief Computes the results of the DBSCAN algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DBSCAN-ALGORITHM">DBSCAN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of DBSCAN, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the DBSCAN algorithm
 *      - \ref InputId  Identifiers of input objects for the DBSCAN algorithm
 *      - \ref ResultId Identifiers of results of the DBSCAN algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::dbscan::Input InputType;
    typedef algorithms::dbscan::Parameter ParameterType;
    typedef algorithms::dbscan::Result ResultType;

    /**
     *  Main constructor
     *  \param[in] epsilon         Radius of neighborhood
     *  \param[in] minObservations Minimal total weight of observations in neighborhood of core observation
     */
    Batch(algorithmFPType epsilon, size_t minObservations);

    /**
     * Constructs a DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Destructor.
     * Releases dynamically allocated resources held by the object.
     */
    ~Batch() { delete _par; }

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
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the results of the DBSCAN algorithm
     * \return Structure that contains the results of the DBSCAN algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated  memory  to store the results of the DBSCAN algorithm
     * \param[in] result  Structure to store the results of the DBSCAN algorithm
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_in, _par, (int)method);
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

} // namespace dbscan
} // namespace algorithms
} // namespace daal
#endif
