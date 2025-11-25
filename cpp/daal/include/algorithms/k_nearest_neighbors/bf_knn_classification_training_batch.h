/* file: bf_knn_classification_training_batch.h */
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
//  Implementation of the interface for k-Nearest Neighbor (kNN) model-based training in the batch processing mode
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_TRAINING_BATCH_H__
#define __BF_KNN_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_types.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup bf_knn_classification_batch Batch
 * @ingroup bf_knn_classification_training
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for BF kNN model-based training using algorithmFPType precision arithmetic
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for BF kNN model-based training with a specified environment in the batch processing mode
     * \param[in] daalEnv   Environment object
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~BatchContainer();

    /**
     * Computes the result of BF kNN model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Provides methods for BF kNN model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">k-Nearest Neighbors algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for BF kNN model-based training, double or float
 * \tparam method           BF kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref bf_knn_classification::interface1::Model "bf_knn_classification::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef algorithms::bf_knn_classification::training::Input InputType;
    typedef algorithms::bf_knn_classification::Parameter ParameterType;
    typedef algorithms::bf_knn_classification::training::Result ResultType;

    /**
     * Default constructor
     */
    Batch();

    /**
     * Constructs a BF kNN training algorithm by copying input objects and parameters
     * of another BF kNN training algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Constructs a BF kNN training algorithm with nClasses parameter
     * \param[in] nClasses   number of classes
     */
    Batch(size_t nClasses);

    /** Destructor */
    ~Batch() { delete _par; }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    virtual ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    virtual const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
     * Get input objects for the BF kNN training algorithm
     * \return %Input objects for the BF kNN training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return static_cast<InputType *>(_in); }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of BF kNN model-based training
     * \return Structure that contains the result of BF kNN model-based training
     */
    ResultPtr getResult() { return Result::cast(_result); }

    /**
     * Resets the results of BF kNN model training algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to a newly allocated BF kNN training algorithm
     * with a copy of the input objects and parameters for this BF kNN training algorithm
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

public:
    InputType input; /*!< %Input objects of the algorithm */

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        const ResultPtr res = getResult();
        DAAL_CHECK(_result, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>((classifier::training::InputIface *)(_in), (ParameterType *)_par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _result.reset(new ResultType());
        _in = &input;
    }

private:
    Batch & operator=(const Batch &);
};

/** @} */
} // namespace interface1

using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
