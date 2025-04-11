/* file: gbt_regression_model_builder.h */
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
//  Implementation of class defining gradient boosting trees regression model builder.
//--
*/
#ifndef __GBT_REGRESSION_MODEL_BUILDER_H__
#define __GBT_REGRESSION_MODEL_BUILDER_H__

#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
/**
 * @defgroup gbt_classification Gradient Boosted Trees Classification
 * \copydoc daal::algorithms::gbt::regression
 * @ingroup regression
 */
/**
 * \brief Contains classes for the gradient boosted trees regression algorithm
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup gbt_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__MODEL_BUILDER"></a>
 * \brief %Model Builder class for gradient boosted trees regression model.
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 */
class DAAL_EXPORT ModelBuilder
{
public:
    /**
     * \brief %Node identifier type is size_t
     */
    typedef size_t NodeId;

    /**
     * \brief %Tree identifier type is size_t
     */
    typedef size_t TreeId;

    static const NodeId noParent = static_cast<size_t>(-1); /*!< %Reserved value for root nodes */

    /**
     * Empty constructor for deserialization
     */
    ModelBuilder();

    /**
     * Constructs the gradient boosted trees regression model builder
     * \param[in] nIterations  Number of trees in model and iterations performed on training stage of gbt algorithm
     * \param[in] nFeatures    Number of features in training dataset
     */
    ModelBuilder(size_t nFeatures, size_t nIterations)
    {
        _status |= initialize(nFeatures, nIterations);
        services::throwIfPossible(_status);
    }

    /**
    *  Create certain tree in the gradient boosted trees regression model
    *  \param[in] nNodes  Number of nodes in created tree
    *  \return Tree identifier
    */
    TreeId createTree(size_t nNodes)
    {
        TreeId resId;
        _status |= createTreeInternal(nNodes, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Leaf node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] response        Response value for leaf node to be predicted
    *  \param[in] cover           Cover of the node (sum_hess)
    *  \return Node identifier
    */
    NodeId addLeafNode(TreeId treeId, NodeId parentId, size_t position, double response, double cover)
    {
        NodeId resId;
        _status |= addLeafNodeInternal(treeId, parentId, position, response, cover, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Split node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] featureIndex    Feature index for splitting
    *  \param[in] featureValue    Feature value for splitting
    *  \param[in] defaultLeft     Behaviour in case of missing values
    *  \param[in] cover           Cover of the node (sum_hess)
    *  \return Node identifier
    */
    NodeId addSplitNode(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft, double cover)
    {
        NodeId resId;
        _status |= addSplitNodeInternal(treeId, parentId, position, featureIndex, featureValue, defaultLeft, cover, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Get built model
    *  \return Model pointer
    */
    ModelPtr getModel()
    {
        _status |= convertModelInternal();
        services::throwIfPossible(_status);
        return _model;
    }

    /**
    *  Get status of model building
    *  \return Status
    */
    services::Status getStatus() const { return _status; }

protected:
    ModelPtr _model;
    services::Status _status;
    services::Status initialize(size_t nFeatures, size_t nIterations);
    services::Status createTreeInternal(size_t nNodes, TreeId & resId);
    services::Status addLeafNodeInternal(TreeId treeId, NodeId parentId, size_t position, double response, double cover, NodeId & res);
    services::Status addSplitNodeInternal(TreeId treeId, NodeId parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft,
                                          double cover, NodeId & res);
    services::Status convertModelInternal();
};
/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
