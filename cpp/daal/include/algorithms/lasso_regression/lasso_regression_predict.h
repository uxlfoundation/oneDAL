/* file: lasso_regression_predict.h */
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
//  Implementation of the interface for lasso regression model-based prediction
//--
*/

#ifndef __LASSO_REGRESSION_PREDICT_H__
#define __LASSO_REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "algorithms/lasso_regression/lasso_regression_predict_types.h"
#include "algorithms/linear_model/linear_model_predict.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup lasso_regression_prediction_batch Batch
 * @ingroup lasso_regression_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the lasso regression model-based prediction
 * <!-- \n<a href="DAAL-REF-LASSOREGRESSION-ALGORITHM">LASSO regression algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for lasso regression model-based prediction
 *                          in the batch processing mode, double or float
 *
 * \par References
 *      - \ref lasso_regression::interface1::Model "lasso_regression::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public linear_model::prediction::Batch<algorithmFPType, linear_model::prediction::defaultDense>
{
public:
    typedef linear_model::prediction::Batch<algorithmFPType, linear_model::prediction::defaultDense> super;

    typedef algorithms::lasso_regression::prediction::Input InputType;
    typedef typename super::ParameterType ParameterType;
    typedef algorithms::lasso_regression::prediction::Result ResultType;

    InputType input; /*!< %Input data structure */

    /** Default constructor */
    Batch() { initialize(); }

    /**
     * Constructs a lasso regression prediction algorithm by copying input objects
     * of another lasso regression prediction algorithm
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
     * Returns the structure that contains the result of lasso regression model-based prediction
     * \return Structure that contains the result of the lasso regression model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(this->_result); }

    /**
     * Returns a pointer to a newly allocated lasso regression prediction algorithm
     * with a copy of the input objects for this lasso regression prediction algorithm
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
        super::initialize();
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
} // namespace lasso_regression
} // namespace algorithms
} // namespace daal

#endif
