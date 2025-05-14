/* file: linear_model_predict.h */
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

#ifndef __LINEAR_MODEL_PREDICT_H__
#define __LINEAR_MODEL_PREDICT_H__

#include "algorithms/linear_model/linear_model_predict_types.h"
#include "algorithms/regression/regression_predict.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup linear_model_prediction_batch Batch
 * @ingroup linear_model_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__BATCHCONTAINER"></a>
 *  \brief Class containing computation methods for the regression model-based prediction
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for the regression model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     *  Computes the result of the regression model-based prediction
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the regression model-based prediction
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the regression model-based prediction
 *                          in the batch processing mode, double or float
 * \tparam method           Computation method in the batch processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for the regression model-based prediction
 *
 * \par References
 *      - \ref linear_model::interface1::Model "linear_model::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public regression::prediction::Batch
{
public:
    typedef algorithms::linear_model::prediction::Input InputType;
    typedef algorithms::linear_model::Parameter ParameterType;
    typedef algorithms::linear_model::prediction::Result ResultType;

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains the result of the regression model-based prediction
     * \return Structure that contains the result of the regression model-based prediction
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

protected:
    void initialize();
};
/** @} */
} // namespace interface1
using interface1::Batch;
using interface1::BatchContainer;
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
