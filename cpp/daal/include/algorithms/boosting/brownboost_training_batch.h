/* file: brownboost_training_batch.h */
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
//  Implementation of the interface for BrownBoost model-based training
//--
*/

#ifndef __BROWN_BOOST_TRAINING_BATCH_H__
#define __BROWN_BOOST_TRAINING_BATCH_H__

#include "algorithms/boosting/brownboost_training_types.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace training
{
namespace interface2
{
/**
 * @defgroup brownboost_training_batch Batch
 * @ingroup brownboost_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of BrownBoost model-based training.
 *        This class is associated with daal::algorithms::brownboost::training::Batch class
*
 * \tparam algorithmFPType  Data type to use in intermediate computations for BrownBoost, double or float
 * \tparam method           BrownBoost model training method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for BrownBoost model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of BrownBoost model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__TRAINING__BATCH"></a>
 * \brief Trains model of the BrownBoost algorithms in the batch processing mode
 * <!-- \n<a href="DAAL-REF-BROWNBOOST-ALGORITHM">BrownBoost algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for BrownBoost, double or float
 * \tparam method           BrownBoost computation method, \ref daal::algorithms::brownboost::training::Method
 *
 * \par Enumerations
 *      - \ref Method                         BrownBoost training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the BrownBoost training algorithm
 *      - \ref classifier::training::ResultId Identifiers of BrownBoost training results
 *
 * \par References
 *      - \ref interface2::Model "Model" class
 *      - classifier::training::Input class
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::brownboost::Parameter ParameterType;
    typedef algorithms::brownboost::training::Result ResultType;

    InputType input; /*!< %Input data structure */

    DAAL_DEPRECATED Batch();

    /**
     * Constructs an BrownBoost training algorithm with nClasses parameter
     * \param[in] nClasses   number of classes
    */
    DAAL_DEPRECATED Batch(size_t nClasses);

    /**
     * Constructs a BrownBoost training algorithm by copying input objects and parameters
     * of another BrownBoost training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() { delete _par; }

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
     * Get input objects for the BrownBoost training algorithm
     * \return %Input objects for the BrownBoost training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of BrownBoost training
     * \return Structure that contains results of BrownBoost training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated BrownBoost training algorithm with a copy of input objects
     * and parameters of this BrownBoost training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(_result, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, method);
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
} // namespace interface2
using interface2::Batch;
using interface2::BatchContainer;

} // namespace training
} // namespace brownboost
} // namespace algorithms
} // namespace daal
#endif // __BROWN_BOOST_TRAINING_BATCH_H__
