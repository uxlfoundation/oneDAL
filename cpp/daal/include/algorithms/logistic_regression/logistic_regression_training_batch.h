/* file: logistic_regression_training_batch.h */
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
//  Implementation of the interface for logistic regression model-based training
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAINING_BATCH_H__
#define __LOGISTIC_REGRESSION_TRAINING_BATCH_H__

#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/logistic_regression/logistic_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace interface3
{
/**
 * @defgroup logistic_regression_training_batch Batch
 * @ingroup logistic_regression_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of logistic regression model-based training.
 *        This class is associated with daal::algorithms::logistic_regression::training::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           logistic regression model training method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for logistic regression model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of logistic regression model-based training in the batch processing mode
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    services::Status setupCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__BATCH"></a>
 * \brief Trains model of the logistic regression algorithms in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_REGRESSION-ALGORITHM">logistic regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for logistic regression, double or float
 * \tparam method           logistic regression computation method, \ref daal::algorithms::logistic_regression::training::Method
 *
 * \par Enumerations
 *      - \ref Method                         logistic regression training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the logistic regression training algorithm
 *      - \ref classifier::training::ResultId Identifiers of logistic regression training results
 *
 * \par References
 *      - \ref logistic_regression::interface1::Model "Model" class
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;
    typedef optimization_solver::iterative_solver::BatchPtr SolverPtr;

    typedef typename super::InputType InputType;
    typedef algorithms::logistic_regression::training::Parameter ParameterType;
    typedef algorithms::logistic_regression::training::Result ResultType;

    InputType input; /*!< %Input data structure */

    /**
     * Constructs the logistic regression training algorithm
     * \param[in] nClasses  Number of classes
     * \param[in] solver    Optimization solver
     */
    Batch(size_t nClasses, const SolverPtr & solver = SolverPtr());

    /**
     * Constructs a logistic regression training algorithm by copying input objects and parameters
     * of another logistic regression training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /** Destructor */
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
     * Get input objects for the logistic regression training algorithm
     * \return %Input objects for the logistic regression training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of logistic regression training
     * \return Structure that contains results of logistic regression training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated logistic regression training algorithm with a copy of input objects
     * and parameters of this logistic regression training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, &parameter(), method);
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
} // namespace interface3
using interface3::BatchContainer;
using interface3::Batch;

} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif // __LOGISTIC_REGRESSION_TRAINING_BATCH_H__
