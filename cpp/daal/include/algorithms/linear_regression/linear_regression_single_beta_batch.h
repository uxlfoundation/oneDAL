/* file: linear_regression_single_beta_batch.h */
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
//  Interface of the linear regression single beta quality metric in the batch processing mode.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_BATCH_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/linear_regression/linear_regression_single_beta_types.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
/**
 * \brief Contains classes for computing linear regression quality metrics for single beta
 */
namespace single_beta
{
namespace interface1
{
/**
 * @defgroup linear_regression_quality_metric_single_beta_batch Batch
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute regression quality metric
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the quality metric, double or float
 * \tparam method           Computation method for the metric, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /** Default constructor */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of linear regression model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLE_BETA__BATCH"></a>
 * \brief Computes the linear regression quality metric in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of metric, double or float
 * \tparam method           Computation method for the metric, \ref Method
 *
 * \par Enumerations
 *      - \ref Method         Computation method for the metric
 *      - \ref DataInputId    Identifiers of input objects for the metric algorithm
 *      - \ref ResultId       Result identifiers for the metric algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::quality_metric::Batch
{
public:
    typedef algorithms::linear_regression::quality_metric::single_beta::Input InputType;
    typedef algorithms::linear_regression::quality_metric::single_beta::Parameter ParameterType;
    typedef algorithms::linear_regression::quality_metric::single_beta::Result ResultType;

    InputType input;         /*!< %Input objects of the algorithm */
    ParameterType parameter; /*!< Parameters of the algorithm */

    /** Default constructor */
    Batch();

    /**
     * Constructs an algorithm by copying input objects and parameters
     * of another algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of the algorithm
     * \return Structure that contains results of the algorithm
     */
    ResultPtr getResult() const { return _result; }

    /**
     * Registers user-allocated memory to store results of the algorithm
     * \param[in] result  Structure to store results of the algorithm
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
     * Registers user-allocated memory to store the input object for the algorithm
     * \param[in] other  Structure to store the input object for the algorithm
     */
    virtual void setInput(const algorithms::Input * other) DAAL_C11_OVERRIDE
    {
        const InputType * inputPtr = static_cast<const InputType *>(other);
        input.set(expectedResponses, inputPtr->get(expectedResponses));
        input.set(predictedResponses, inputPtr->get(predictedResponses));
        input.set(model, inputPtr->get(model));
    }

    /**
     * Returns a pointer to the newly allocated algorithm with a copy of input objects
     * and parameters of this algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    virtual algorithms::ResultPtr getResultImpl() const DAAL_C11_OVERRIDE { return _result; }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace single_beta
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
