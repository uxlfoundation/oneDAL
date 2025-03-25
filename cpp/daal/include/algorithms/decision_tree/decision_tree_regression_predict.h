/* file: decision_tree_regression_predict.h */
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
//  Implementation of the interface for Decision tree regression model-based prediction
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_H__
#define __DECISION_TREE_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"
#include "algorithms/regression/regression_predict.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
namespace interface2
{
/**
 * @defgroup decision_tree_regression_prediction_batch Batch
 * @ingroup decision_tree_regression_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__BATCHCONTAINER"></a>
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
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__BATCH"></a>
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
 *      - \ref decision_tree::regression::interface1::Model "decision_tree::regression::Model" class
 *      - \ref training::interface2::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public algorithms::regression::prediction::Batch
{
public:
    typedef algorithms::regression::prediction::Batch super;

    typedef algorithms::decision_tree::regression::prediction::Input InputType;
    typedef algorithms::decision_tree::regression::Parameter ParameterType;
    typedef algorithms::decision_tree::regression::prediction::Result ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Batch();

    /**
     * Constructs a Decision tree prediction algorithm by copying input objects and parameters
     * of another Decision tree prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual algorithms::regression::prediction::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of decision tree model-based prediction
     * \return Structure that contains the result of the decision tree model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

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
        services::Status s = getResult()->template allocate<algorithmFPType>(_in, &parameter, (int)method);
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

} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
