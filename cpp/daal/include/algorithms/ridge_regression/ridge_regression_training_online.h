/* file: ridge_regression_training_online.h */
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
//  Implementation of the interface for ridge regression model-based training in the online processing mode
//--
*/

#ifndef __RIDGE_REGRESSION_TRAINING_ONLINE_H__
#define __RIDGE_REGRESSION_TRAINING_ONLINE_H__

#include "algorithms/algorithm.h"
#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "algorithms/linear_model/linear_model_training_online.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup ridge_regression_online Online
 * @ingroup ridge_regression_training
 * @{
 */
/**
 * \brief Class containing methods for ridge regression model-based training
 * in the online processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public TrainingContainerIface<online>
{
public:
    /**
     * Constructs a container for ridge regression model-based training with a specified environment in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of ridge regression model-based training in the online processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of ridge regression model-based training in the online processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__ONLINE"></a>
 * \brief Provides methods for ridge regression model-based training in the online processing mode
 * <!-- \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for
 *                          ridge regression model-based training , double or float
 * \tparam method           Ridge regression training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref interface1::TrainParameter "TrainParameter" class
 *      - \ref ridge_regression::interface1::Model "ridge_regression::Model" class
 *      - \ref ridge_regression::interface1::ModelNormEq "ridge_regression::ModelNormEq" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = normEqDense>
class DAAL_EXPORT Online : public linear_model::training::Online
{
public:
    typedef algorithms::ridge_regression::training::Input InputType;
    typedef algorithms::ridge_regression::TrainParameter ParameterType;
    typedef algorithms::ridge_regression::training::Result ResultType;
    typedef algorithms::ridge_regression::training::PartialResult PartialResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Training parameters */

    /** Default constructor */
    Online();
    /**
     * Constructs a ridge regression training algorithm by copying input objects and parameters of another ridge regression training algorithm in the
     * online processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects and parameters of the algorithm
     */
    Online(const Online<algorithmFPType, method> & other);

    ~Online() {}

    virtual regression::training::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains a partial result of ridge regression model-based training
     * \return Structure that contains a partial result of ridge regression model-based training
     */
    PartialResultPtr getPartialResult() { return PartialResultType::cast(_partialResult); }

    /**
     * Returns the structure that contains the result of ridge regression model-based training
     * \return Structure that contains the result of ridge regression model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Returns a pointer to a newly allocated ridge regression training algorithm with a copy of the input objects and parameters of this ridge
     * regression training algorithm in the online processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online<algorithmFPType, method> > clone() const { return services::SharedPtr<Online<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Online<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Online<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(&input, &parameter, method);
        _res               = _result.get();
        return s;
    }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getPartialResult()->template allocate<algorithmFPType>(&input, &parameter, method);
        _pres              = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getPartialResult()->template initialize<algorithmFPType>(&input, &parameter, method);
        _pres              = _partialResult.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(online, OnlineContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _partialResult.reset(new PartialResultType());
        _result.reset(new ResultType());
    }

private:
    Online & operator=(const Online &);
}; // class : Online
/** @} */
} // namespace interface1

using interface1::OnlineContainer;
using interface1::Online;

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
