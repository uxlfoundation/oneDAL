/* file: decision_tree_classification_predict.h */
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
//  Implementation of the interface for Decision tree classification model-based prediction
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_PREDICT_H__
#define __DECISION_TREE_CLASSIFICATION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/decision_tree/decision_tree_classification_predict_types.h"
#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "algorithms/classifier/classifier_predict.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace prediction
{
namespace interface2
{
/**
 * @defgroup decision_tree_classification_prediction_batch Batch
 * @ingroup decision_tree_classification_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for Decision tree model-based prediction
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for Decision tree model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    /**
     *  Computes the result of Decision tree model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the Decision tree model-based prediction
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for Decision tree model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for Decision tree model-based prediction
 *
 * \par References
 *      - \ref decision_tree::classification::interface1::Model "decision_tree::classification::Model" class
 *      - \ref training::interface2::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::decision_tree::classification::prediction::Input InputType;
    typedef algorithms::decision_tree::classification::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Batch(size_t nClasses = 2);

    /**
     * Constructs a Decision tree prediction algorithm by copying input objects and parameters
     * of another Decision tree prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Get input objects for the Decision tree prediction algorithm
     * \return %Input objects for the Decision tree prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated Decision tree prediction algorithm with a copy of input objects
     * of this Decision tree prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in  = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
    }

private:
    Batch & operator=(const Batch &);
};

/** @} */
} // namespace interface2

using interface2::BatchContainer;
using interface2::Batch;

} // namespace prediction
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
