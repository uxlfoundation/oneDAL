/* file: gbt_classification_predict.h */
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
//  Implementation of the interface for gradient boosted trees classification model-based prediction
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_H__
#define __GBT_CLASSIFICATION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
/**
 * \brief Contains classes for prediction based on models
 */
namespace prediction
{
/**
 * @defgroup gbt_classification_prediction_batch Batch
 * @ingroup gbt_classification_prediction
 * @{
 */
/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the gradient boosted trees algorithm.
 *        This class is associated with daal::algorithms::gbt::prediction::interface1::Batch class
 *        and supports method to compute gbt prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           gradient boosted trees computation method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for gradient boosted trees model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of gbt model-based prediction
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__PREDICTION__BATCH"></a>
 * \brief Predicts gradient boosted trees classification results
 * <!-- \n<a href="DAAL-REF-gbt-ALGORITHM">gbt algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the gbt algorithm, double or float
 * \tparam method           gradient boosted trees computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       gbt prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *                                                          for the gradient boosted trees prediction algorithm
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

    typedef algorithms::gbt::classification::prediction::Input InputType;
    typedef algorithms::gbt::classification::prediction::Parameter ParameterType;
    typedef algorithms::gbt::classification::prediction::Result ResultType;

    InputType input; /*!< %Input objects of the algorithm */

    /**
     * Constructs gradient boosted trees prediction algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses);

    /**
     * Constructs a gradient boosted trees prediction algorithm by copying input objects and parameters
     * of another gradient boosted trees prediction algorithm
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
     * Gets input objects for the gradient boosted trees prediction algorithm
     * \return %Input objects for the Gradient Boosted Trees prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of model-based prediction
     * \return Structure that contains the result of the model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Returns a pointer to the newly allocated gradient boosted trees prediction algorithm with a copy of input objects
     * and parameters of this gradient boosted trees prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(&input, _par, 0);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
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
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
