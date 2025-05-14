/* file: decision_forest_classification_model_builder.h */
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
//  Implementation of class defining decision_forest classification model builder.
//--
*/
#ifndef __DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER_H__
#define __DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER_H__

#include "algorithms/decision_forest/decision_forest_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
/**
 * @defgroup decision_forest_classification Decision Forest Classification
 * \copydoc daal::algorithms::decision_forest::classification
 * @ingroup classification
 */
/**
 * \brief Contains classes for the decision_forest classification algorithm
 */
namespace classification
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_CLASSIFICATION_MODEL_BUILDER"></a>
 * \brief %Model Builder class for Decision Forest Classification Model algorithm
 *
 * \par References
 *      - \ref classification::interface1::Model "classification::Model" class
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
     * Constructs the Decision forest classification model builder
     * \param[in] nClasses  Number of classes
     * \param[in] nTrees  Number of trees in model
     */
    ModelBuilder(const size_t nClasses, const size_t nTrees) : _nClasses(nClasses)
    {
        _status |= initialize(nClasses, nTrees);
        services::throwIfPossible(_status);
    }

    /**
     * Empty constructor for deserialization
     */
    ModelBuilder();

    /**
    *  Create certain tree in the decision forest model
    *  \param[in] nNodes  Number of nodes in created tree
    *  \return Tree identifier
    */
    TreeId createTree(const size_t nNodes)
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
    *  \param[in] classLabel      Class label to be predicted
    *  \param[in] cover           Cover (Hessian sum) of the node
    *  \return Node identifier
    */
    NodeId addLeafNode(const TreeId treeId, const NodeId parentId, const size_t position, const size_t classLabel, const double cover)
    {
        NodeId resId;
        _status |= addLeafNodeInternal(treeId, parentId, position, classLabel, cover, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    /**
    *  Create Leaf node and add it to certain tree
    *  \param[in] treeId          Tree to which new node is added
    *  \param[in] parentId        Parent node to which new node is added (use noParent for root node)
    *  \param[in] position        Position in parent (e.g. 0 for left and 1 for right child in a binary tree)
    *  \param[in] proba           Array with probability values for each class
    *  \param[in] cover           Cover (Hessian sum) of the node
    *  \return Node identifier
    */
    NodeId addLeafNodeByProba(const TreeId treeId, const NodeId parentId, const size_t position, const double * const proba, const double cover)
    {
        NodeId resId;
        _status |= addLeafNodeByProbaInternal(treeId, parentId, position, proba, cover, resId);
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
    *  \param[in] cover           Cover (Hessian sum) of the node
    *  \return Node identifier
    */
    NodeId addSplitNode(const TreeId treeId, const NodeId parentId, const size_t position, const size_t featureIndex, const double featureValue,
                        const int defaultLeft, const double cover)
    {
        NodeId resId;
        _status |= addSplitNodeInternal(treeId, parentId, position, featureIndex, featureValue, defaultLeft, cover, resId);
        services::throwIfPossible(_status);
        return resId;
    }

    void setNFeatures(size_t nFeatures)
    {
        if (!_model.get())
        {
            _status |= services::ErrorNullModel;
            services::throwIfPossible(_status);
        }
        else
        {
            _model->setNFeatures(nFeatures);
        }
    }

    /**
    *  Get built model
    *  \return Model pointer
    */
    ModelPtr getModel()
    {
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
    services::Status initialize(const size_t nClasses, const size_t nTrees);
    services::Status createTreeInternal(const size_t nNodes, TreeId & resId);
    services::Status addLeafNodeInternal(const TreeId treeId, const NodeId parentId, const size_t position, const size_t classLabel,
                                         const double cover, NodeId & res);
    services::Status addLeafNodeByProbaInternal(const TreeId treeId, const NodeId parentId, const size_t position, const double * const proba,
                                                const double cover, NodeId & res);
    services::Status addSplitNodeInternal(const TreeId treeId, const NodeId parentId, const size_t position, const size_t featureIndex,
                                          const double featureValue, const int defaultLeft, const double cover, NodeId & res);

private:
    size_t _nClasses;
};
/** @} */
} // namespace interface2
using interface2::ModelBuilder;

} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
