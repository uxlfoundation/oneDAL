/* file: multiclass_confusion_matrix_batch.h */
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
//  Interface for the multi-class confusion matrix algorithm in the batch processing mode
//--
*/

#ifndef __MULTICLASS_CONFUSION_MATRIX_BATCH_H__
#define __MULTICLASS_CONFUSION_MATRIX_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/multiclass_confusion_matrix_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace multiclass_confusion_matrix
{
namespace interface1
{
/**
 * @defgroup quality_metric_multiclass_batch Batch
 * @ingroup quality_metric_multiclass
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute the confusion matrix for the multi-class classifier
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the multi-class confusion matrix, double or float
 * \tparam method           Method for computing the multi-class confusion matrix, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the multi-class confusion matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the multi-class confusion matrix algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTICLASS_CONFUSION_MATRIX__BATCH"></a>
 * \brief Computes the confusion matrix for a multi-class classifier in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the multi-class confusion matrix, double or float
 * \tparam method           Method for computing the multi-class confusion matrix, \ref Method
 *
 * \par Enumerations
 *      - \ref Method             Method for computing the multi-class confusion matrix
 *      - \ref InputId              Identifiers of input objects for the multi-class confusion matrix algorithm
 *      - \ref ResultId             Result identifiers for the multi-class confusion matrix algorithm
 *      - \ref MultiClassMetricsId  Identifiers of resulting metrics associated with the multi-class confusion matrix
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::quality_metric::Batch
{
public:
    typedef algorithms::classifier::quality_metric::multiclass_confusion_matrix::Input InputType;
    typedef algorithms::classifier::quality_metric::multiclass_confusion_matrix::Parameter ParameterType;
    typedef algorithms::classifier::quality_metric::multiclass_confusion_matrix::Result ResultType;

    InputType input;         /*!< %Input objects of the algorithm */
    ParameterType parameter; /*!< Parameters of the algorithm */

    /**
     * Default constructor
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses = 2);

    /**
     * Constructs a confusion matrix algorithm by copying input objects and parameters
     * of another confusion matrix algorithm
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
     * Returns the structure that contains results of the multi-class confusion matrix algorithm
     * \return Structure that contains results of the multi-class confusion matrix algorithm
     */
    ResultPtr getResult() const { return _result; }

    /**
     * Registers user-allocated memory to store results of the multi-class confusion matrix algorithm
     * \param[in] result  Structure to store results of the multi-class confusion matrix algorithm
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store input of the multi-class confusion matrix algorithm
     * \param[in] other  Structure to store results of the multi-class confusion matrix algorithm
     */
    virtual void setInput(const algorithms::Input * other) DAAL_C11_OVERRIDE
    {
        const InputType * inputPtr = static_cast<const InputType *>(other);
        input.set(predictedLabels, inputPtr->get(predictedLabels));
        input.set(groundTruthLabels, inputPtr->get(groundTruthLabels));
    }

    /**
     * Returns a pointer to the newly allocated confusion matrix algorithm with a copy of input objects
     * and parameters of this confusion matrix algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
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

} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
