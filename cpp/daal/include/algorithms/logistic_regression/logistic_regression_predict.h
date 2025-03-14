/* file: logistic_regression_predict.h */
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
//  Implementation of the interface for logistic regression model-based prediction
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_H__
#define __LOGISTIC_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "algorithms/logistic_regression/logistic_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
/**
 * \brief Contains classes for prediction based on models
 */
namespace prediction
{
/**
 * @defgroup logistic_regression_prediction_batch Batch
 * @ingroup logistic_regression_prediction
 * @{
 */

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the logistic regression algorithm.
 *        This class is associated with daal::algorithms::logistic_regression::prediction::interface2::Batch class
 *        and supports method to compute logistic regression prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           logistic regression computation method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for logistic regression model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of logistic regression model-based prediction
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__PREDICTION__BATCH"></a>
 * \brief Predicts logistic regression results
 * <!-- \n<a href="DAAL-REF-LOGISTIC-REGRESSION-ALGORITHM">logistic regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the logistic regression algorithm, double or float
 * \tparam method           logistic regression computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       logistic regression prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *                                                          for the logistic regression prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects of the algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of prediction results
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref classifier::prediction::interface1::Input "classifier::prediction::Input" class
 *      - \ref classifier::prediction::interface2::Result "classifier::prediction::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::logistic_regression::prediction::Input InputType;
    typedef algorithms::classifier::Parameter ParameterType;
    typedef algorithms::classifier::prediction::Result ResultType;

    /**
     * Constructs logistic regression prediction algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses);

    /**
     * Constructs a logistic regression prediction algorithm by copying input objects and parameters
     * of another logistic regression prediction algorithm
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
     * Gets input objects for the logistic regression prediction algorithm
     * \return %Input objects for the logistic regression prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated logistic regression prediction algorithm with a copy of input objects
     * and parameters of this logistic regression prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
    * Returns the structure that contains the result of the logistic regression model-based prediction
    * \return Structure that contains the result of the logistic regression model-based prediction
    */
    classifier::prediction::ResultPtr getResult() { return ResultType::cast(_result); }

public:
    InputType input; /*!< %Input objects of the algorithm */

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = static_cast<ResultType *>(_result.get())->allocate<algorithmFPType>(&input, _par, 0);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, interface2::BatchContainer, algorithmFPType, method)(&_env);
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2
using interface2::BatchContainer;
using interface2::Batch;

} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
