/* file: decision_forest_regression_model.h */
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
//  Implementation of the class defining the decision forest regression model
//--
*/

#ifndef __DECISION_FOREST_REGRESSION_MODEL_H__
#define __DECISION_FOREST_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/regression/regression_model.h"
#include "algorithms/regression/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_regression.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
/**
 * @defgroup decision_forest_regression Decision Forest Regression
 * \copydoc daal::algorithms::decision_forest::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for decision forest regression algorithm
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup decision_forest_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_REGRESSION_MODEL"></a>
 * \brief %Base class for models trained with the decision forest regression algorithm
 *
 * \tparam modelFPType  Data type to store decision forest model data, double or float
 *
 * \par References
 *      - \ref regression::training::interface2::Batch "regression::training::Batch" class
 *      - \ref regression::prediction::interface1::Batch "regression::prediction::Batch" class
 */
class DAAL_EXPORT Model : public algorithms::regression::Model
{
public:
    DECLARE_MODEL(Model, algorithms::regression::Model);

    /**
    *  Get number of trees in the decision forest model
    *  \return number of trees
    *  \DAAL_DEPRECATED_USE{ Model::getNumberOfTrees }
    */
    virtual size_t numberOfTrees() const = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
    */
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
    */
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const = 0;

    /**
     *  Removes all trees from the model
     */
    virtual void clear() = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const = 0;

    /**
    *  Get number of trees in the decision forest model
    *  \return number of trees
    */
    virtual size_t getNumberOfTrees() const = 0;

protected:
    Model();
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

/** @} */
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
