/* file: linear_model_model.h */
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

#ifndef __LINEAR_MODEL_MODEL_H__
#define __LINEAR_MODEL_MODEL_H__

#include "algorithms/algorithm_types.h"
#include "algorithms/regression/regression_model.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup linear_model Linear Model
 * \brief Contains base classes of regression algorithms with linear model
 * @ingroup regression
 */
/**
 * \brief Contains classes of the regression algorithm
 */
namespace linear_model
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup linear_model
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__PARAMETER"></a>
 * \brief Parameters for the regression algorithm
 *
 * \snippet linear_model/linear_model_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter();
    Parameter(const Parameter & other);
    Parameter & operator=(const Parameter & other);
    bool interceptFlag; /*!< Flag that indicates whether the intercept needs to be computed */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__MODEL"></a>
 * \brief %Base class for models trained with the regression algorithm
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public regression::Model
{
public:
    DAAL_CAST_OPERATOR(Model)

    virtual ~Model() {}

    /**
     * Initializes ridge regression coefficients of the regression model
     */
    virtual services::Status initialize() = 0;

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    virtual size_t getNumberOfBetas() const = 0;

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    virtual size_t getNumberOfResponses() const = 0;

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    virtual bool getInterceptFlag() const = 0;

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    virtual data_management::NumericTablePtr getBeta() = 0;
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
 *
 * \return Status of computations
 */
DAAL_EXPORT services::Status checkModel(linear_model::Model * model, const daal::algorithms::Parameter & par, size_t nBeta, size_t nResponses);

} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
