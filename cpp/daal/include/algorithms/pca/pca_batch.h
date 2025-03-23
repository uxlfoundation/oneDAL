/* file: pca_batch.h */
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
//  Implementation of the interface for the PCA algorithm in the batch processing mode
//--
*/

#ifndef __PCA_BATCH_H__
#define __PCA_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
/**
    * \brief Contains version 3.0 of oneAPI Data Analytics Library interface.
    */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public AnalysisContainerIface<batch>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the PCA algorithm in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};
/**
 * @defgroup pca_batch Batch
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHCONTAINER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the PCA algorithm in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCH"></a>
 * \brief Computes the results of the PCA algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for PCA, double or float
 * \tparam method           PCA computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = correlationDense>
class DAAL_EXPORT Batch : public Analysis<batch>
{
public:
    typedef algorithms::pca::Input InputType;
    typedef algorithms::pca::BatchParameter<algorithmFPType, method> ParameterType;
    typedef algorithms::pca::Result ResultType;

    /** Default constructor */
    Batch();

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return method; };

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure for storing the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains final results of the PCA algorithm
     * \return Structure that contains final results of the PCA
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    InputType input;                                   /*!< Input data structure */
    BatchParameter<algorithmFPType, method> parameter; /*!< Parameters */

protected:
    ResultPtr _result;

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};

/** @} */
} // namespace interface3
using interface3::BatchContainer;
using interface3::Batch;

} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
