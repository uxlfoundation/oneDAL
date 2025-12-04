/* file: bf_knn_classification_predict.h */
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
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_H__
#define __BF_KNN_CLASSIFICATION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup bf_knn_classification_prediction_batch Batch
 * @ingroup bf_knn_classification_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Class containing computation methods for BF kNN model-based prediction
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for BF kNN model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    /**
     *  Computes the result of BF kNN model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the BF kNN model-based prediction
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for BF kNN model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for BF kNN model-based prediction
 *
 * \par References
 *      - \ref bf_knn_classification::interface1::Model "bf_knn_classification::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::bf_knn_classification::prediction::Input InputType;
    typedef algorithms::bf_knn_classification::Parameter ParameterType;
    typedef algorithms::bf_knn_classification::prediction::Result ResultType;

    /**
     * Default constructor
     */
    Batch();

    /**
     * Constructs a BF kNN prediction algorithm by copying input objects and parameters
     * of another BF kNN prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Constructs a BF kNN prediction algorithm with nClasses parameter
     * \param[in] nClasses   number of classes
     */
    Batch(size_t nClasses);

    /**
     * Destructor.
     * Releases dynamically allocated resources held by the object.
     */
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
     * Registers user-allocated memory to store the results of the BF kNN prediction algorithm
     * \param[in] result  Structure to store the results of the BF kNN prediction algorithm
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the results of the BF kNN prediction algorithm
     * \return Structure that contains the results of the BF kNN prediction algorithm
     */
    ResultPtr getResult() { return Result::cast(_result); }

    /**
     * Get input objects for the BF kNN prediction algorithm
     * \return %Input objects for the BF kNN prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return static_cast<InputType *>(_in); }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated BF kNN prediction algorithm with a copy of input objects
     * of this BF kNN prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

public:
    InputType input; /*!< %Input objects of the algorithm */

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = static_cast<ResultType *>(_result.get())->allocate<algorithmFPType>(&input, _par, (int)method);
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

} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
