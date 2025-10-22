/* file: kmeans_init_batch.h */
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
//  Implementation of the interface for initializing K-Means algorithm
//  in the batch processing mode
//--
*/

#ifndef __KMEANS_INIT_BATCH_H__
#define __KMEANS_INIT_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/kmeans/kmeans_init_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface2
{
/**
 * @defgroup kmeans_init_batch Batch
 * @ingroup kmeans_init
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of initialization of K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::init::Batch class
 *        and supports the method of computing initial clusters for K-Means algorithm in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref daal::algorithms::kmeans::init::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes initial values for K-Means algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__BATCHBASE"></a>
 *  \brief Base class representing K-Means algorithm initialization in the batch processing mode
 */
class DAAL_EXPORT BatchBase : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::kmeans::init::Input InputType;
    typedef algorithms::kmeans::init::Parameter ParameterType;
    typedef algorithms::kmeans::init::Result ResultType;

    /** Default destructor */
    virtual ~BatchBase() {}

protected:
    BatchBase() {}

    explicit BatchBase(ParameterType * parameter) { _par = parameter; }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__BATCH"></a>
 * \brief Computes initial clusters for K-Means algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm initialization description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Methods of computing initial clusters for K-Means algorithm
 *      - \ref InputId  Identifiers of input objects for computing initial clusters for K-Means algorithm
 *      - \ref ResultId Identifiers of results of computing initial clusters for K-Means algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public BatchBase
{
public:
    /**
     *  Main constructor
     *  \param[in] nClusters   Number of clusters
     */
    Batch(size_t nClusters);

    /**
     * Constructs an algorithm that computes initial clusters for K-Means algorithm
     * by copying input objects and parameters of another algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /** Destructor */
    ~Batch() DAAL_C11_OVERRIDE { delete _par; }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the results of computing initial clusters for K-Means algorithm
     * \return Structure that contains the results of computing initial clusters for K-Means algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store the results of computing initial clusters for K-Means algorithm
     * \param[in] result  Structure to store the results of computing initial clusters for K-Means algorithm
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated algorithm that computes initial clusters for K-Means algorithm
     * with a copy of input objects and parameters of this algorithm
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
    }

public:
    ParameterType & parameter; /*!< %Parameters */
    InputType input;           /*!< %Input data structure */

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2
using interface2::BatchContainer;
using interface2::BatchBase;
using interface2::Batch;
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
#endif
