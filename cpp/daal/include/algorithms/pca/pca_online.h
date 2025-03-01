/* file: pca_online.h */
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
//  Implementation of the interface for the PCA algorithm in the online processing mode
//--
*/

#ifndef __PCA_ONLINE_H__
#define __PCA_ONLINE_H__

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
namespace interface1
{
/**
 * @defgroup pca_online Online
 * @ingroup pca
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public AnalysisContainerIface<online>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_CORRELATIONDENSE_CPU"></a>
 * \brief Class containing methods to compute the result of the PCA algorithm
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, correlationDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINECONTAINER_ALGORITHMFPTYPE_SVDDENSE_CPU"></a>
 * \brief Class containing methods to compute the results of the PCA algorithm
 */
template <typename algorithmFPType, CpuType cpu>
class OnlineContainer<algorithmFPType, svdDense, cpu> : public AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the PCA algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of the PCA algorithm in the online processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the PCA algorithm in the online processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE"></a>
 * \brief Computes the results of the PCA algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = correlationDense>
class Online : public Analysis<online>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Computes the results of the PCA Correlation algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template <typename algorithmFPType>
class Online<algorithmFPType, correlationDense> : public Analysis<online>
{
public:
    typedef algorithms::pca::Input InputType;
    typedef algorithms::pca::OnlineParameter<algorithmFPType, correlationDense> ParameterType;
    typedef algorithms::pca::Result ResultType;
    typedef algorithms::pca::PartialResult<correlationDense> PartialResultType;

    /** Default constructor */
    Online();

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, correlationDense> & other);

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return (int)correlationDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<correlationDense> > & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<correlationDense> > getPartialResult() { return _partialResult; }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, correlationDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, correlationDense> >(cloneImpl());
    }

    InputType input;                                              /*!< Input data structure */
    OnlineParameter<algorithmFPType, correlationDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<correlationDense> > _partialResult;
    ResultPtr _result;

    virtual Online<algorithmFPType, correlationDense> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Online<algorithmFPType, correlationDense>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, correlationDense);
        _res               = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, correlationDense);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(&input, &parameter, correlationDense);
        _pres              = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, correlationDense)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<correlationDense>());
        _result.reset(new ResultType());
    }

private:
    Online & operator=(const Online &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE_ALGORITHMFPTYPE_SVDDENSE"></a>
 * \brief Computes the results of the PCA SVD algorithm
 * <!-- \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the PCA algorithm, double or float
 */
template <typename algorithmFPType>
class Online<algorithmFPType, svdDense> : public Analysis<online>
{
public:
    typedef algorithms::pca::Input InputType;
    typedef algorithms::pca::OnlineParameter<algorithmFPType, svdDense> ParameterType;
    typedef algorithms::pca::Result ResultType;
    typedef algorithms::pca::PartialResult<svdDense> PartialResultType;

    /** Default constructor */
    Online();

    /**
     * Constructs a PCA algorithm by copying input objects and parameters of another PCA algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, svdDense> & other);

    ~Online() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    int getMethod() const DAAL_C11_OVERRIDE { return (int)svdDense; }

    /**
     * Registers user-allocated  memory to store the results of the PCA algorithm
     * \param[in] partialResult    Structure for storing partial result of the PCA algorithm
     */
    services::Status setPartialResult(const services::SharedPtr<PartialResult<svdDense> > & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store the results of the PCA algorithm
     * \param[in] res    Structure to store the results of the PCA algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains partial results of the PCA algorithm
     * \return Structure that contains partial results of the PCA algorithm
     */
    services::SharedPtr<PartialResult<svdDense> > getPartialResult() { return _partialResult; }

    /**
     * Returns structure that contains the results of the PCA algorithm
     * \return Structure that contains the results of the PCA algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated PCA algorithm
     * with a copy of input objects and parameters of this PCA algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, svdDense> > clone() const
    {
        return services::SharedPtr<Online<algorithmFPType, svdDense> >(cloneImpl());
    }

    InputType input;                                      /*!< Input data structure */
    OnlineParameter<algorithmFPType, svdDense> parameter; /*!< Parameters */

protected:
    services::SharedPtr<PartialResult<svdDense> > _partialResult;
    ResultPtr _result;

    virtual Online<algorithmFPType, svdDense> * cloneImpl() const DAAL_C11_OVERRIDE { return new Online<algorithmFPType, svdDense>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(_pres, &parameter, svdDense);
        _res               = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, svdDense);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(&input, &parameter, svdDense);
        _pres              = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, svdDense)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResult<svdDense>());
        _result.reset(new ResultType());
    }

private:
    Online & operator=(const Online &);
};
/** @} */
} // namespace interface1
using interface1::OnlineContainer;
using interface1::Online;

} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
