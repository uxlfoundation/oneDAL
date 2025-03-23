/* file: stump_regression_predict.h */
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
//  Implementation of the interface for decision stump model-based prediction
//--
*/

#ifndef __STUMP_REGRESSION_PREDICT_H__
#define __STUMP_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/regression/regression_predict.h"
#include "algorithms/stump/stump_regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace regression
{
/**
 * @defgroup stump_regression_prediction_batch Batch
 * @ingroup stump_regression_prediction
 * @{
 */
/**
 * \brief Contains classes to make prediction based on the decision stump model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__REGRESSION__PREDICTION__METHOD"></a>
 * Available methods to make prediction based on the decision stump model
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the decision stump prediction algorithm.
 *        It is associated with the daal::algorithms::stump::regression::prediction::interface1::Batch class
 *        and supports methods to run based on the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision stump prediction algorithm, double or float
 * \tparam method           Decision stump model-based prediction method, \ref Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for decision stump model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision stump model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__PREDICTION__BATCH"></a>
 * \brief Predicts results of the decision stump regression
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision stump prediction algorithm, double or float
 * \tparam method           Decision stump model-based prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       Decision stump prediction methods
 *      - \ref daal::algorithms::regression::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *                                                          for the decision stump prediction algorithm
 *      - \ref daal::algorithms::regression::prediction::ModelInputId         Identifiers of input %Model objects
 *                                                          for the decision stump prediction algorithm
 *      - \ref daal::algorithms::regression::prediction::ResultId             Identifiers of the results of the decision stump prediction algorithm
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref daal::algorithms::regression::prediction::interface1::Input "regression::prediction::Input" class
 *      - \ref daal::algorithms::regression::prediction::interface1::Result "regression::prediction::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::regression::prediction::Batch
{
public:
    typedef daal::algorithms::regression::prediction::Batch super;

    typedef algorithms::stump::regression::prediction::Input InputType;
    typedef algorithms::stump::regression::Parameter ParameterType;
    typedef algorithms::stump::regression::prediction::Result ResultType;

    InputType input; /*!< %Input data structure */

    Batch();

    /**
     * Constructs decision stump regression algorithm by copying input objects
     * of another decision stump regression algorithm
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
     * Get input objects for the decision stump prediction algorithm
     * \return %Input objects for the decision stump prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of stump model-based prediction
     * \return Structure that contains the result of the stump model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Returns a pointer to the newly allocated decision stump regression algorithm
     * with a copy of input objects and parameters of this decision stump regression algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(_in, _par, (int)method);
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
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace prediction
/** @} */
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif
