/* file: linear_regression_predict.h */
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
//  Implementation of the interface for linear regression model-based prediction
//--
*/

#ifndef __LINEAR_REGRESSION_PREDICT_H__
#define __LINEAR_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "algorithms/linear_regression/linear_regression_predict_types.h"
#include "algorithms/linear_model/linear_model_predict.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup linear_regression_prediction_batch Batch
 * @ingroup linear_regression_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the linear regression model-based prediction
 * <!-- \n<a href="DAAL-REF-LINEARREGRESSION-ALGORITHM">Linear regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for linear regression model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for linear regression model-based prediction
 *
 * \par References
 *      - \ref linear_regression::interface1::Model "linear_regression::Model" class
 *      - \ref linear_regression::interface1::ModelNormEq "linear_regression::ModelNormEq" class
 *      - \ref linear_regression::interface1::ModelQR "linear_regression::ModelQR" class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the linear regression model-based prediction
 * <!-- \n<a href="DAAL-REF-LINEARREGRESSION-ALGORITHM">Linear regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for linear regression model-based prediction
 *                          in the batch processing mode, double or float
 *
 * \par References
 *      - \ref linear_regression::interface1::Model "linear_regression::Model" class
 *      - \ref linear_regression::interface1::ModelNormEq "linear_regression::ModelNormEq" class
 *      - \ref linear_regression::interface1::ModelQR "linear_regression::ModelQR" class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 */
template <typename algorithmFPType>
class DAAL_EXPORT Batch<algorithmFPType, defaultDense>
    : public linear_model::prediction::Batch<algorithmFPType, linear_model::prediction::defaultDense>
{
public:
    typedef algorithms::linear_regression::prediction::Input InputType;
    typedef algorithms::linear_regression::Parameter ParameterType;
    typedef algorithms::linear_regression::prediction::Result ResultType;

    InputType input; /*!< %Input data structure */

    /** Default constructor */
    Batch() { initialize(); }

    /**
     * Constructs a linear regression prediction algorithm by copying input objects
     * of another linear regression prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  of the algorithm
     */
    Batch(const Batch<algorithmFPType, defaultDense> & other) : input(other.input) { initialize(); }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)defaultDense; }

    /**
     * Returns the structure that contains the result of linear regression model-based prediction
     * \return Structure that contains the result of the linear regression model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(this->_result); }

    /**
     * Returns a pointer to a newly allocated linear regression prediction algorithm
     * with a copy of the input objects for this linear regression prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, defaultDense> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, defaultDense> >(cloneImpl());
    }

    virtual regression::prediction::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

protected:
    virtual Batch<algorithmFPType, defaultDense> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, defaultDense>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = getResult()->template allocate<algorithmFPType>(this->_in, 0, 0);
        this->_res         = this->_result.get();
        return s;
    }

    void initialize()
    {
        linear_model::prediction::Batch<algorithmFPType, linear_model::prediction::defaultDense>::initialize();
        this->_in = &input;
        this->_result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace prediction
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
