/* file: stump_regression_model.h */
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
//  Implementation of the class defining the decision stump model
//--
*/

#ifndef __STUMP_REGRESSION_MODEL_H__
#define __STUMP_REGRESSION_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/matrix.h"
#include "algorithms/regression/regression_model.h"
#include "algorithms/decision_tree/decision_tree_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for Decision Stump algorithm
 */
namespace stump
{
/**
 * @defgroup stump_regression Decision Stump for Regression
 * \copydoc daal::algorithms::stump::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for decision stump regression algorithm
 */
namespace regression
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__REGRESSION__VARIABLE_IMPORTANCE_MODE"></a>
 * \brief Variable importance computation mode
 */
enum VariableImportanceMode
{
    none, /* Do not compute */
    MDI,
    MDA_Raw,
    MDA_Scaled
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup stump_regression
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__STUMP__REGRESSION__PARAMETER"></a>
 * \brief Stump algorithm parameters
 *
 * \snippet stump/stump_regression_model.h Parameter source code
 */
/* [Parameter source code] */
// CLARIFICATION:: Added parameter class to support different split criterions for stump.
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Main constructor
     */
    Parameter() : daal::algorithms::Parameter(), varImportance(none) {}

    /**
     * Checks a parameter of the Decision tree algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    VariableImportanceMode varImportance; /*!< Variable importance mode.
                                               Variable importance computation is not supported for current version of the library */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__MODEL"></a>
 * \brief %Model of the regression trained by the stump::regression::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::decision_tree::regression::Model
{
public:
    DECLARE_MODEL_IFACE(Model, decision_tree::regression::Model);

    /**
     * Constructs the decision stump model
     * \tparam modelFPType  Data type to store decision stump model data, double or float
     * \param[out] stat      Status of the model construction
     * \return Decision stump model
     */
    static services::SharedPtr<Model> create(services::Status * stat = NULL);

    /**
     * Default constructor for Model to creator
     */
    Model() {}

    virtual ~Model();

    /**
     *  Returns the split feature
     *  \return Index of the feature over which the split is made
     */
    size_t getSplitFeature() const;

    /**
     *  Returns a value of the feature that defines the split
     *  \return Value of the feature over which the split is made
     */
    template <typename modelFPType>
    DAAL_EXPORT modelFPType getSplitValue();

    /**
     *  Returns an average of the weighted responses for the "left" subset
     *  \return Average of the weighted responses for the "left" subset
     */
    template <typename modelFPType>
    DAAL_EXPORT modelFPType getLeftValue();

    /**
     *  Returns an average of the weighted responses for the "right" subset
     *  \return Average of the weighted responses for the "right" subset
     */
    template <typename modelFPType>
    DAAL_EXPORT modelFPType getRightValue();

protected:
    Model(services::Status & st);

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;
};

typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/** @} */
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif
