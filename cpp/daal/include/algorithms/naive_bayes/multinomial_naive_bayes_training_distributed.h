/* file: multinomial_naive_bayes_training_distributed.h */
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
//  Implementation of the interface for multinomial naive Bayes model-based training
//  in the distributed processing mode
//--
*/

#ifndef __NAIVE_BAYES_TRAINING_DISTRIBUTED_H__
#define __NAIVE_BAYES_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/naive_bayes/multinomial_naive_bayes_training_types.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface2
{
/**
 * @defgroup multinomial_naive_bayes_training_distributed Distributed
 * @ingroup multinomial_naive_bayes_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDCONTAINER"></a>
 *  \brief Class containing methods to compute naive Bayes training results in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the naive Bayes training algorithm in the distributed processing mode,
 *                          double or float
 * \tparam method           Naive Bayes training method on the first step in distributed processing mode, \ref Method
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train naive Bayes in the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for multinomial naive Bayes model-based training with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of naive Bayes model-based training
     * in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    /**
     * Computes the result of naive Bayes model-based training
     * in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training naive Bayes model in the distributed processing mode
 *  <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for multinomial naive Bayes training, double or float
 *  \tparam method           Computation method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method %Training methods for the naive Bayes algorithm
 *
 */
template <ComputeStep step, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Distributed
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training Naive Bayes partial model in the distributed processing mode
 *  <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the multinomial naive Bayes training on the first step in distributed
 *                           processing mode, double or float
 *  \tparam method           Naive Bayes training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method %Training methods for the multinomial naive Bayes on the first step in the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public Online<algorithmFPType, method>
{
public:
    typedef Online<algorithmFPType, method> super;

    typedef typename super::InputType InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType ResultType;
    typedef typename super::PartialResultType PartialResultType;

    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Distributed(size_t nClasses) : Online<algorithmFPType, method>::Online(nClasses) {}

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other) : Online<algorithmFPType, method>(other) {}

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

private:
    Distributed & operator=(const Distributed &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTED_STEP2MASTER_ALGORITHMFPTTYPE_METHOD"></a>
 *  \brief Algorithm class for training naive Bayes final model on the second step in the distributed processing mode
 *  <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the multinomial naive Bayes training on the second step in
 *                           distributed processing mode, double or float
 *  \tparam method           Naive Bayes training method on the second step in distributed processing mode, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method %Training methods for the multinomial naive Bayes algorithm
 *
 */
template <typename algorithmFPType, Method method>
class Distributed<step2Master, algorithmFPType, method> : public Training<distributed>
{
public:
    typedef algorithms::multinomial_naive_bayes::training::DistributedInput InputType;
    typedef algorithms::multinomial_naive_bayes::Parameter ParameterType;
    typedef algorithms::multinomial_naive_bayes::training::Result ResultType;
    typedef algorithms::multinomial_naive_bayes::training::PartialResult PartialResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the distributed training algorithm */
    InputType input;         /*!< %Input objects of the algorithm */

    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Distributed(size_t nClasses);

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Master, algorithmFPType, method> & other);
    virtual ~Distributed() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Registers user-allocated memory for storing partial training results
     * \param[in] partialResult    Structure for storing partial results
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult);
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Registers user-allocated memory to store results of Naive Bayes training
     * \param[in] result  Structure to store  results of Naive Bayes training
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
     * Returns the structure that contains results of Naive Bayes training
     * \return Structure that contains results of Naive Bayes training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Validates parameters of the finalizeCompute() method
     *
     * \return Status of computations
     */
    services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        PartialResultPtr partialResult = getPartialResult();
        DAAL_CHECK(partialResult, services::ErrorNullResult);
        services::Status s;
        DAAL_CHECK_STATUS(s, partialResult->check(_par, method));
        ResultPtr result = getResult();
        DAAL_CHECK(result, services::ErrorNullResult);
        DAAL_CHECK_STATUS(s, result->check(_pres, _par, method));
        return s;
    }

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    PartialResultPtr _partialResult;
    ResultPtr _result;

    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        PartialResultPtr pres = getPartialResult();
        ResultPtr res         = getResult();
        services::Status s    = res->template allocate<algorithmFPType>(pres.get(), &parameter, (int)method);
        _res                  = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        PartialResultPtr pres = getPartialResult();
        services::Status s    = pres->template allocate<algorithmFPType>((classifier::training::InputIface *)(&input), &parameter, (int)method);
        _pres                 = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE { return services::Status(); }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
        _partialResult.reset(new PartialResultType());
    }

private:
    Distributed & operator=(const Distributed &);
};
/** @} */
} // namespace interface2
using interface2::DistributedContainer;
using interface2::Distributed;

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} //namespace daal
#endif
