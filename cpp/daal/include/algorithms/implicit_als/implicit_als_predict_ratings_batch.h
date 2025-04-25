/* file: implicit_als_predict_ratings_batch.h */
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
//  Implementation of the interface for implicit ALS model-based ratings prediction
//  in the batch processing mode
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_BATCH_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/implicit_als/implicit_als_predict_ratings_types.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
/**
 * @defgroup implicit_als_prediction_batch Batch
 * @ingroup implicit_als_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the implicit ALS ratings prediction algorithm in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for implicit ALS model-based ratings prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of implicit ALS model-based ratings prediction
     * in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__BATCH"></a>
 *  \brief Predicts the results of the implicit ALS algorithm
 * <!-- \n<a href="DAAL-REF-IMPLICIT_ALS-ALGORITHM">Implicit ALS algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 *  \tparam method           Implicit ALS prediction method, \ref Method
 *
 *  \par Enumerations
 *      - \ref Method Implicit ALS prediction methods
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Prediction
{
public:
    typedef algorithms::implicit_als::prediction::ratings::Input InputType;
    typedef algorithms::implicit_als::Parameter ParameterType;
    typedef algorithms::implicit_als::prediction::ratings::Result ResultType;

    InputType input;         /*!< Input objects for the algorithm */
    ParameterType parameter; /*!< \ref implicit_als::interface1::Parameter "Parameters" of the ratings prediction algorithm */

    /**
     * Default constructor
     */
    Batch();

    /**
     * Constructs an implicit ALS ratings prediction algorithm by copying input objects and parameters
     * of another implicit ALS ratings prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

    /**
     * Returns the structure that contains the computed prediction results
     * \return Structure that contains the computed prediction results
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory for storing the prediction results
     * \param[in] result Structure for storing the prediction results
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated ALS ratings prediction algorithm with a copy of input objects
     * of this ALS ratings prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    ResultPtr _result;

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
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
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
#endif
