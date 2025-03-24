/* file: adaboost_predict.h */
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
//  Implementation of the interface for AdaBoost model-based prediction
//--
*/

#ifndef __ADA_BOOST_PREDICT_H__
#define __ADA_BOOST_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/boosting/adaboost_model.h"
#include "algorithms/boosting/adaboost_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
/**
 * \brief Contains classes for making prediction based on the AdaBoost models
 */
namespace prediction
{
/**
 * @defgroup adaboost_prediction_batch Batch
 * @ingroup adaboost_prediction
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__PREDICTION__METHOD"></a>
 * Available methods for making predictions based on AdaBoost model
 */
enum Method
{
    defaultDense = 0,            /*!< Default method */
    samme        = defaultDense, /*!< SAMME algorithm */
    sammeR       = 1             /*!< SAMME.R algorithm, need probabilities from weak learner prediction result */
};

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the AdaBoost algorithm.
 *        It is associated with daal::algorithms::adaboost::prediction::interface1::Batch class
 *        and supports method to compute AdaBoost prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the AdaBoost, double or float
 * \tparam method           AdaBoost computation method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for AdaBoost model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of AdaBoost model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__PREDICTION__BATCH"></a>
 * \brief Predict AdaBoost classification results
 * <!-- \n<a href="DAAL-REF-ADABOOST-ALGORITHM">AdaBoost algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the AdaBoost, double or float
 * \tparam method           AdaBoost computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       Enumeration of supported AdaBoost prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Enumeration of supported Numeric Table input arguments
 *                                                          of the AdaBoost prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Enumeration of supported Model input arguments of the AdaBoost prediction algorithm
 *      - \ref classifier::prediction::ResultId             Enumeration of supported AdaBoost prediction results
 *
 * \par References
 *      - \ref interface2::Model "Model" class
 *      - classifier::prediction::Input class
 *      - \ref classifier::prediction::interface2::Result "classifier::prediction::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::adaboost::prediction::Input InputType;
    typedef algorithms::adaboost::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    InputType input; /*!< %Input objects of the algorithm */

    /**
     * Constructs AdaBoost prediction algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses);

    /**
     * Constructs an AdaBoost prediction algorithm by copying input objects and parameters
     * of another AdaBoost prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

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
     * Get input objects for the AdaBoost prediction algorithm
     * \return %Input objects for the AdaBoost prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated AdaBoost prediction algorithm with a copy of input objects
     * and parameters of this AdaBoost prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, _par, 0);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
    }

private:
    Batch & operator=(const Batch &);
};
} // namespace interface2
using interface2::Batch;
using interface2::BatchContainer;

/** @} */
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal
#endif
