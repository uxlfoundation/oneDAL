/* file: regression_predict.h */
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
//  Implementation of the interface for the regression model-based prediction
//--
*/

#ifndef __REGRESSION_PREDICT_H__
#define __REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/regression/regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup base_regression_prediction_batch Batch
 * @ingroup base_regression_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the regression model-based prediction
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
class Batch : public daal::algorithms::Prediction
{
public:
    typedef algorithms::regression::prediction::Input InputType;
    typedef algorithms::Parameter ParameterType;
    typedef algorithms::regression::prediction::Result ResultType;

    virtual ~Batch() {}
    virtual InputType * getInput() = 0;

    /**
     * Registers user-allocated memory to store the result of the regression model-based prediction
     * \param[in] res    Structure to store the result of the regression model-based prediction
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the result of the regression model-based prediction
     * \return Structure that contains the result of the the regression model-based prediction
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated regression prediction algorithm with a copy of input objects
     * and parameters of this regression prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const { return services::SharedPtr<Batch>(cloneImpl()); }

protected:
    ResultPtr _result;
    virtual Batch * cloneImpl() const override = 0;
};
/** @} */
} // namespace interface1
using interface1::Batch;
} // namespace prediction
} // namespace regression
} // namespace algorithms
} // namespace daal
#endif
