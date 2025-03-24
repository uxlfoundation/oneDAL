/* file: multinomial_naive_bayes_predict.h */
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
//  Implementation of the interface for multinomial naive Bayes model-based prediction
//  in the batch processing mode
//--
*/

#ifndef __NAIVE_BAYES_PREDICT_H__
#define __NAIVE_BAYES_PREDICT_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/naive_bayes/multinomial_naive_bayes_predict_types.h"
#include "algorithms/classifier/classifier_predict.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
/*
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface
 */
namespace interface2
{
/**
 * @defgroup multinomial_naive_bayes_prediction_batch Batch
 * @ingroup multinomial_naive_bayes_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__BATCHCONTAINER"></a>
 * \brief Runs the prediction based on the multinomial naive Bayes model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for prediction based on the multinomial naive Bayes model, double or float
 * \tparam method           Multinomial naive Bayes prediction method, \ref Method
 */
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for multinomial naive Bayes model-based prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of multinomial naive Bayes model-based prediction in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__BATCH"></a>
 *  \brief Predicts the results of the multinomial naive Bayes classification
 *  <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for prediction based on the multinomial naive Bayes model, double or float
 *  \tparam method           Multinomial naive Bayes prediction method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Multinomial naive Bayes prediction methods
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, prediction::Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef algorithms::multinomial_naive_bayes::prediction::Input InputType;
    typedef algorithms::multinomial_naive_bayes::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    InputType input;         /*!< %Input objects of the algorithm */
    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the prediction algorithm */
    /**
     * Default constructor
     * \param nClasses  Number of classes
     */
    Batch(size_t nClasses);

    /**
     * Constructs multinomial naive Bayes prediction algorithm by copying input objects and parameters
     * of another multinomial naive Bayes prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

    /**
     * Get input objects for the multinomial naive Bayes prediction algorithm
     * \return %Input objects for the multinomial naive Bayes prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated multinomial naive Bayes prediction algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes prediction algorithm
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
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
