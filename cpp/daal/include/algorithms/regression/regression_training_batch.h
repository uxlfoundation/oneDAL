/* file: regression_training_batch.h */
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
//  Implementation of the interface for the regression model-based training
//  in the batch processing mode
//--
*/

#ifndef __REGRESSION_TRAINING_BATCH_H__
#define __REGRESSION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/regression/regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup base_regression_training_batch Batch
 * @ingroup base_regression_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__BATCH"></a>
 * \brief Provides methods for the regression model-based training in the batch processing mode
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Batch : public Training<batch>
{
public:
    typedef algorithms::regression::training::Input InputType;
    typedef algorithms::regression::training::Result ResultType;

    virtual ~Batch() {}
    /**
     * Get input objects for the regression model-based training algorithm
     * \return Input objects for the regression model-based training algorithm
     */
    virtual InputType * getInput() = 0;

    /**
     * Registers user-allocated memory to store the result of the regression model-based training
     * \param[in] res    Structure to store the result of the regression model-based training
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

    /* Resets the results of the regression model-based training
     * \return Status of the operation
     */
    virtual services::Status resetResult() = 0;

    /**
     * Returns a pointer to the newly allocated regression training algorithm with a copy of input objects
     * and parameters of this regression training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const { return services::SharedPtr<Batch>(cloneImpl()); }

    /**
     * Returns the structure that contains the result of the regression model-based training
     * \return Structure that contains the result of the regression model-based training
     */
    ResultPtr getResult() { return _result; }

protected:
    virtual Batch * cloneImpl() const override = 0;
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;
} // namespace training
} // namespace regression
} // namespace algorithms
} // namespace daal
#endif
