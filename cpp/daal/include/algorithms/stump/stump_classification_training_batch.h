/* file: stump_classification_training_batch.h */
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
//  Implementation of the interface for decision stump model-based training
//  in the batch processing mode
//--
*/

#ifndef __STUMP_CLASSIFICATION_TRAINING_BATCH_H__
#define __STUMP_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/stump/stump_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup stump_classification_training_batch Batch
 * @ingroup stump_classification_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the the decision stump training algorithm.
 *        It is associated with the daal::algorithms::stump::classification::training::Batch class
 *        and supports methods to train the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the the decision stump training algorithm, double or float
 * \tparam method           the decision stump training method, \ref Method
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for decision stump model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision stump model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Trains the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the the decision stump training method, double or float
 * \tparam method           Decision stump training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                         Decision stump training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the decision stump training algorithm
 *      - \ref classifier::training::ResultId Identifiers of results of the decision stump training algorithm
 *
 * \par References
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 *      - \ref interface1::Model "Model" class
 *      - Result class
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::stump::classification::Parameter ParameterType;
    typedef algorithms::stump::classification::training::Result ResultType;

    InputType input; /*!< %Input data structure */

    DAAL_DEPRECATED Batch(size_t nClasses = 2);

    /**
     * Constructs decision stump training algorithm by copying input objects
     * of another decision stump training algorithm
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
     * Get input objects for the stump training algorithm
     * \return %Input objects for the stump training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains computed results of the decision stump training algorithm
     * \return Structure that contains computed results of the decision stump training algorithm
     */
    training::ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult)
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated decision stump training algorithm
     * with a copy of input objects and parameters of this decision stump training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        training::ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
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
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif // __stump__classification_TRAINING_BATCH_H__
