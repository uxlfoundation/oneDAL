/* file: elastic_net_training_batch.h */
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
//  Implementation of the interface for elastic net model-based training in the batch processing mode
//--
*/

#ifndef __ELASTIC_NET_TRAINING_BATCH_H__
#define __ELASTIC_NET_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/elastic_net/elastic_net_training_types.h"
#include "algorithms/elastic_net/elastic_net_model.h"
#include "algorithms/linear_model/linear_model_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
namespace interface1
{
/**
 * @defgroup elastic_net_batch Batch
 * @ingroup elastic_net_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for normal equations elastic net model-based training using algorithmFPType precision arithmetic
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for elastic net model-based training with a specified environment in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~BatchContainer();

    /**
     * Computes the result of elastic net model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__TRAINING__BATCH"></a>
 * \brief Provides methods for elastic net model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ELASTICNET-ALGORITHM">Elastic net algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for elastic net model-based training, double or float
 * \tparam method           Elastic net training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref elastic_net::interface1::Model "elastic_net::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public linear_model::training::Batch
{
public:
    typedef algorithms::elastic_net::training::Input InputType;
    typedef optimization_solver::iterative_solver::BatchPtr SolverPtr;
    typedef algorithms::elastic_net::training::Parameter ParameterType;
    typedef algorithms::elastic_net::training::Result ResultType;

    InputType input; /*!< %Input data structure */

    /** Default constructor */
    Batch(const SolverPtr & solver = SolverPtr());

    /**
     * Constructs a elastic net training algorithm by copying input objects
     * and parameters of another elastic net training algorithm in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

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
     * Get input objects for the elastic net training algorithm
     * \return %Input objects for the elastic net training algorithm
     */
    virtual regression::training::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of elastic net model-based training
     * \return Structure that contains the result of elastic net model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the results of elastic net model-based training
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to a newly allocated elastic net training algorithm
     * with a copy of the input objects and parameters for this elastic net training algorithm
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(&input, static_cast<const ParameterType *>(_par), method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1

using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal

#endif
