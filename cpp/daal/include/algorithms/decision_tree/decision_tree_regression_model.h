/* file: decision_tree_regression_model.h */
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
//  Implementation of the class defining the Decision tree regression model
//--
*/

#ifndef __DECISION_TREE_REGRESSION_MODEL_H__
#define __DECISION_TREE_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/decision_tree/decision_tree_model.h"
#include "algorithms/regression/regression_model.h"
#include "algorithms/regression/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_regression.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for Decision tree algorithm
 */
namespace decision_tree
{
/**
 * @defgroup decision_tree_regression Decision Tree for Regression
 * \copydoc daal::algorithms::decision_tree::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for decision tree regression algorithm
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup decision_tree_regression
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_TREE__REGRESSION__PARAMETER"></a>
 * \brief Decision tree algorithm parameters
 *
 * \snippet decision_tree/decision_tree_regression_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Main constructor
     */
    Parameter() : daal::algorithms::Parameter(), pruning(reducedErrorPruning), maxTreeDepth(0), minObservationsInLeafNodes(5) {}

    /**
     * Checks a parameter of the Decision tree algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    Pruning pruning;                   /*!< Pruning method for Decision tree */
    size_t maxTreeDepth;               /*!< Maximum tree depth. 0 means unlimited depth. */
    size_t minObservationsInLeafNodes; /*!< Minimum number of observations in the leaf node. Can be any positive number. */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the Decision tree algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface2::Batch "training::Batch" class
 *      - \ref prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public algorithms::regression::Model
{
public:
    DECLARE_MODEL_IFACE(Model, algorithms::regression::Model);

    /**
     * Constructs the model for decision tree regression
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model();

    virtual ~Model();

    class ModelImpl;
    typedef services::SharedPtr<ModelImpl> ModelImplPtr;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl * impl() const { return _impl.get(); }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl * impl() { return _impl.get(); }

    /**
     * Constructs the model of decision tree algorithm
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(services::Status * stat = NULL);

    /**
     * \copydoc regression::Model::getNumberOfFeatures
     */
    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    *  Perform Depth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
    */
    void traverseDF(algorithms::regression::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Breadth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
    */
    void traverseBF(algorithms::regression::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Depth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    void traverseDFS(tree_utils::regression::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Breadth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    void traverseBFS(tree_utils::regression::TreeNodeVisitor & visitor) const;

protected:
    Model(services::Status & st);

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    ModelImplPtr _impl; /*!< Model implementation */
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
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
