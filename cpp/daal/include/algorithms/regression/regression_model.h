/* file: regression_model.h */
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
//  Implementation of the class defining the regression model
//--
*/

#ifndef __REGRESSION_MODEL_H__
#define __REGRESSION_MODEL_H__

#include "algorithms/model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup regression Regression
 * \brief Contains classes for work with the regression algorithms
 * @ingroup training_and_prediction
 */
/**
 * @defgroup base_regression Base Regression
 * \copydoc daal::algorithms::regression
 * @ingroup regression
 */
/**
 * \brief Contains base classes for the regression algorithms
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup base_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the regression algorithm
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    DAAL_CAST_OPERATOR(Model)

    virtual ~Model() {}

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    virtual size_t getNumberOfFeatures() const = 0;
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;
} // namespace regression
} // namespace algorithms
} // namespace daal
#endif
