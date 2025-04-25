/* file: multi_class_classifier_train.h */
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
//  Implementation of the interface for multi-class classifier model-based training
//  in the batch processing mode
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_TRAIN_H__
#define __MULTI_CLASS_CLASSIFIER_TRAIN_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
/**
 * \brief Contains classes for training the multi-class classifier model
 */
namespace training
{
/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @defgroup multi_class_classifier_training_batch Batch
 * @ingroup multi_class_classifier_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute the results of multi-class classifier model-based training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the multi-class classifier, double or float
 * \tparam method           Computation method of the algprithm, \ref daal::algorithms::multi_class_classifier::training::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for multi-class classifier model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of multi-class classifier model-based training in the batch processing mode
     *
     * \return Status of computation
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__BATCH"></a>
 * \brief Algorithm for the multi-class classifier model training
 * <!-- \n<a href="DAAL-REF-MULTICLASSCLASSIFIER-ALGORITHM">Multi-class classifier algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the multi-class classifier training algorithm, double or float
 * \tparam method           Computation method for the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Multi-class classifier training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the multi-class classifier algprithm
 *      - \ref classifier::training::ResultId Identifiers of multi-class classifier training results
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = oneAgainstOne>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::multi_class_classifier::Parameter ParameterType;
    typedef algorithms::multi_class_classifier::training::Result ResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;         /*!< %Input objects of the algorithm */

    /**
     * Default constructor
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED Batch();

    /**
     * Default constructor
     * \param[in] nClasses                         Number of classes
     */
    Batch(size_t nClasses);

    /**
     * Constructs multi-class classifier training algorithm by copying input objects and parameters
     * of another multi-class classifier training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    ~Batch() {}

    /**
     * Get input objects for the multi-class classifier training algorithm
     * \return %Input objects for the multi-class classifier training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the training results of the multi-class classifier algorithm
     * \return Structure that contains the training results of the multi-class classifier algorithm
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
     * Returns a pointer to the newly allocated multi-class classifier training algorithm
     * with a copy of input objects and parameters of this multi-class classifier training algorithm
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2
using interface2::BatchContainer;
using interface2::Batch;

} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} //namespace daal
#endif
