/* file: linear_regression_model.h */
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
//  Implementation of the class defining the linear regression model
//--
*/

#ifndef __LINEAR_REGRESSION_MODEL_H__
#define __LINEAR_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/linear_model/linear_model_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup linear_regression Linear Regression
 * \copydoc daal::algorithms::linear_regression
 * @ingroup linear_model
 */
namespace linear_regression
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup linear_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PARAMETER"></a>
 * \brief Parameters for the linear regression algorithm
 *
 * \snippet linear_regression/linear_regression_model.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public linear_model::Parameter
{};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the linear regression algorithm
 *
 * \tparam modelFPType  Data type to store linear regression model data, double or float
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public linear_model::Model
{
public:
    DAAL_CAST_OPERATOR(Model)
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/**
 * Checks the correctness of linear regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of linear regression algorithm
 * \param[in]  nBeta             Required number of linear regression coefficients
 * \param[in]  nResponses        Required number of responses on the training stage
 * \param[in]  method            Computation method
 *
 * \return Status of computations
 */
DAAL_EXPORT services::Status checkModel(linear_regression::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses,
                                        int method);

} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
