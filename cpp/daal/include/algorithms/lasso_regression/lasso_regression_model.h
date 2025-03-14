/* file: lasso_regression_model.h */
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
//  Implementation of the class defining the lasso regression model
//--
*/

#ifndef __LASSO_REGRESSION_MODEL_H__
#define __LASSO_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/linear_model/linear_model_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup lasso_regression LASSO Regression
 * \copydoc daal::algorithms::lasso_regression
 * @ingroup linear_model
 */
namespace lasso_regression
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the lasso regression algorithm
 *
 * \tparam modelFPType  Data type to store lasso regression model data, double or float
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public linear_model::Model
{
public:
    DECLARE_MODEL(Model, linear_model::Model);
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
} // namespace interface1

using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/**
 * Checks the correctness of lasso regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of lasso regression algorithm
 * \param[in]  nBeta             Required number of lasso regression coefficients
 * \param[in]  nResponses        Required number of responses on the training stage
 * \param[in]  method            Computation method
 *
 * \return Status of computations
 */
DAAL_EXPORT services::Status checkModel(lasso_regression::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses,
                                        int method);

} // namespace lasso_regression
} // namespace algorithms
} // namespace daal

#endif
