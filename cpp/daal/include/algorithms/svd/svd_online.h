/* file: svd_online.h */
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
//  Implementation of the interface for the SVD algorithm in the online processing mode
//--
*/

#ifndef __SVD_ONLINE_H__
#define __SVD_ONLINE_H__

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
 * @defgroup svd_online Online
 * @ingroup svd
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINECONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm in the online processing mode
 *
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public daal::algorithms::AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();
    /**
     * Computes a partial result of the SVD algorithm in the online processing mode
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of the SVD algorithm in the online processing mode
     * \return Status of computations
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINE"></a>
 * \brief Computes results of the SVD algorithm in the online processing mode.
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 * \par Enumerations
 *      - \ref Method   SVD computation methods
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Online : public daal::algorithms::Analysis<online>
{
public:
    typedef OnlinePartialResult PartialResult;
    typedef OnlinePartialResultPtr PartialResultPtr;

    typedef algorithms::svd::Input InputType;
    typedef algorithms::svd::Parameter ParameterType;
    typedef algorithms::svd::Result ResultType;
    typedef algorithms::svd::OnlinePartialResult PartialResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< SVD parameters structure */

    Online();

    /**
     * Constructs an SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, method> & other);

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns the structure that contains computed partial results of the SVD algorithm
     * \return Structure that contains computed partial results of the SVD algorithm
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store computed results of the SVD algorithm
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store computed results of the SVD algorithm
     * \param[in] partialResult    Structure for storing partial results of the SVD algorithm
     * \param[in] initFlag        Flag that specifies whether the partial results are initialized
     * \return Status of computations
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult, bool initFlag = false)
    {
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

    /**
     * Validates parameters of the finalizeCompute() method
     * \return Status of computations
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        if (_partialResult)
        {
            services::Status s = _partialResult->check(_par, method);
            if (!s) return s;
        }
        else
        {
            return services::Status(services::ErrorNullResult);
        }

        if (_result) return _result->check(_partialResult.get(), _par, method);
        return services::Status(services::ErrorNullResult);
    }

    /**
     * Returns a pointer to the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, method> > clone() const { return services::SharedPtr<Online<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Online<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Online<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(_pres, 0, 0);
        _res               = _result.get();
        return s;
    }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult.reset(new PartialResultType());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->initialize<algorithmFPType>(_in, 0, 0);
        _pres              = _partialResult.get();
        return s;
    }

    void initialize()
    {
        Analysis<online>::_ac = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, method)(&_env);
        _in                   = &input;
        _par                  = &parameter;
    }

private:
    PartialResultPtr _partialResult;
    ResultPtr _result;

    Online & operator=(const Online &);
};
/** @} */
} // namespace interface1
using interface1::OnlineContainer;
using interface1::Online;

} // namespace svd
} // namespace algorithms
} // namespace daal
#endif
