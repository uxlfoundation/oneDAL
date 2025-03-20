/* file: implicit_als_predict_ratings_distributed.h */
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
//  Implementation of the interface for implicit ALS model-based ratings prediction
//  in the distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DISTRIBUTED_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
/**
 * @defgroup implicit_als_prediction_distributed Distributed
 * @ingroup implicit_als_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDCONTAINER"></a>
 * \brief Class that contains methods to run implicit ALS model-based prediction in the distributed processing mode
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class that contains methods to run implicit ALS model-based prediction in the first step of
 *        the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public DistributedPredictionContainerIface
{
public:
    /**
     * Constructs a container for implicit ALS model-based ratings prediction with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    ~DistributedContainer();

    /**
     * Computes a partial result of implicit ALS model-based prediction
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of implicit ALS model-based prediction
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTED"></a>
 * \brief Runs implicit ALS model-based prediction in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-IMPLICIT_ALS-ALGORITHM">Implicit ALS algorithm description and usage models</a> -->
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref ComputeStep  Computation steps
 *      - \ref Method       Computation methods
 *
 * \par References
 *      - \ref implicit_als::interface1::Parameter "implicit_als::Parameter" class
 *      - \ref Distributed class
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed : public daal::algorithms::DistributedPrediction
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Performs implicit ALS model-based prediction in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-IMPLICIT_ALS-ALGORITHM">Implicit ALS algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref DistributedInput<step1Local> class
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::DistributedPrediction
{
public:
    typedef algorithms::implicit_als::prediction::ratings::DistributedInput<step1Local> InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::prediction::ratings::Result ResultType;
    typedef algorithms::implicit_als::prediction::ratings::PartialResult PartialResultType;

    DistributedInput<step1Local> input; /*!< %Input data structure */
    ParameterType parameter;            /*!< \ref implicit_als::interface1::Parameter "Parameters" of the algorithm */

    /**
     * Default constructor
     */
    Distributed();

    /**
     * Constructs an implicit ALS ratings prediction algorithm by copying input objects and parameters
     * of another implicit ALS ratings prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other);

    virtual ~Distributed() {}

    /**
     * Returns the structure that contains the results of the implicit ALS ratings prediction algorithm
     * \return Structure that contains the results of the implicit ALS ratings prediction algorithm
     */
    ResultPtr getResult() { return _partialResult->get(finalResult); }

    /**
     * Returns the structure that contains computed partial results of the implicit ALS ratings prediction algorithm
     * \return Structure that contains computed partial results of the implicit ALS ratings prediction algorithm
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory for storing the prediction results
     * \param[in] result Structure for storing the prediction results
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _partialResult->set(finalResult, result);
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory for storing partial results of the implicit ALS ratings prediction algorithm
     * \param[in] partialResult  Structure for storing partial results of the implicit ALS ratings prediction algorithm
     * \param[in] initFlag       Flag that specifies whether partial results are initialized
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult)
        DAAL_CHECK(partialResult->get(finalResult), services::ErrorNullResult)
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated ALS ratings prediction algorithm with a copy of input objects
     * and parameters of this ALS ratings prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    PartialResultPtr _partialResult;

    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _pres              = _partialResult.get();
        return s;
    }

    virtual services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface1
using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
#endif
