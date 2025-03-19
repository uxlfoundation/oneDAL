/* file: linear_regression_training_batch.h */
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
//  Implementation of the interface for linear regression model-based training
//  in the batch processing mode
//--
*/

#ifndef __LINEAR_REGRESSION_TRAINING_BATCH_H__
#define __LINEAR_REGRESSION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "algorithms/linear_regression/linear_regression_model.h"
#include "algorithms/linear_model/linear_model_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup linear_regression_batch Batch
 * @ingroup linear_regression_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for normal equations linear regression
 *        model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for linear regression model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of linear regression model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__BATCH"></a>
 * \brief Provides methods for linear regression model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LINEARREGRESSION-ALGORITHM">Linear regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for linear regression model-based training, double or float
 * \tparam method           Linear regression training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref linear_regression::interface1::Model "linear_regression::Model" class
 *      - \ref linear_regression::interface1::ModelNormEq "linear_regression::ModelNormEq" class
 *      - \ref linear_regression::interface1::ModelQR "linear_regression::ModelQR" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = normEqDense>
class DAAL_EXPORT Batch : public linear_model::training::Batch
{
public:
    typedef algorithms::linear_regression::training::Input InputType;
    typedef algorithms::linear_regression::Parameter ParameterType;
    typedef algorithms::linear_regression::training::Result ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Training \ref interface1::Parameter "parameters" */

    /** Default constructor */
    Batch();

    /**
     * Constructs a linear regression training algorithm by copying input objects
     * and parameters of another linear regression training algorithm in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    ~Batch() {}

    virtual regression::training::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of linear regression model-based training
     * \return Structure that contains the result of linear regression model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the results of linear regression model-based training
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to a newly allocated linear regression training algorithm
     * with a copy of the input objects and parameters for this linear regression training algorithm
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(&input, &parameter, method);
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
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
