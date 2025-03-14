/* file: logistic_regression_model.h */
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
//  Implementation of logistic regression model class
//--
*/

#ifndef __LOGISTIC_REGRESIION_MODEL_H__
#define __LOGISTIC_REGRESIION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup logistic_regression Logistic regression
 * \copydoc daal::algorithms::logistic_regression
 * @ingroup classification
 */
/**
 * \brief Contains classes for the logistic regression algorithm
 */
namespace logistic_regression
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup logistic_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL"></a>
 * \brief %Model of the classifier trained by the logistic_regression::training::Batch algorithm.
 *
 * \par References
 *      - \ref logistic_regression::training::interface3::Batch "training::Batch" class
 *      - \ref logistic_regression::prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
    * Returns the number of regression coefficients
    * \return Number of regression coefficients
    */
    virtual size_t getNumberOfBetas() const = 0;

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

    /**
    * Returns the numeric table that contains regression coefficients
    * \return Table that contains regression coefficients
    */
    virtual const data_management::NumericTablePtr getBeta() const = 0;

protected:
    Model() : classifier::Model() {}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
