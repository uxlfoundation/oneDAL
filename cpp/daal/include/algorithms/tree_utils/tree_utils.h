/* file: tree_utils.h */
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
//  Implementation of the class defining the Decision tree classification model
//--
*/

#ifndef __TREE_UTILS__
#define __TREE_UTILS__

namespace daal
{
namespace algorithms
{
/**
 * @defgroup tree_utils Tree utils
 * \brief Contains classes for work with the tree-based algorithms
 * @ingroup training_and_prediction
 */
namespace tree_utils
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__NODEDESCRIPTOR"></a>
 * \brief %Struct containing base description of node in decision tree
 */
struct DAAL_EXPORT NodeDescriptor
{
    size_t level;            /*!< Number of connections between the node and the root */
    double impurity;         /*!< Measure of the homogeneity of the response variable at the node (i.e., the value of the criterion) */
    size_t nNodeSampleCount; /*!< Number of samples at the node */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__SPLITNODEDESCRIPTOR"></a>
 * \brief %Struct containing description of split node in decision tree
 */
struct DAAL_EXPORT SplitNodeDescriptor : public NodeDescriptor
{
    size_t featureIndex; /*!< Feature used for splitting the node */
    double featureValue; /*!< Threshold value at the node */
    double coverValue;   /*!< Cover, a sum of the Hessian values of the loss function evaluated at the points flowing through the node */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__TREENODEVISITOR"></a>
 * \brief %Interface of abstract visitor used in tree traversal methods.
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "Model" class
 *      - \ref decision_tree::regression::interface1::Model "Model" class
 */
template <typename LeafNodeDescriptorType>
class DAAL_EXPORT TreeNodeVisitor
{
public:
    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] desc     The structure containing description of the leaf visited
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onSplitNode(const SplitNodeDescriptor & desc) = 0;

    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] desc     The structure containing description of the leaf visited
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onLeafNode(const LeafNodeDescriptorType & desc) = 0;
};

} // namespace interface1
using interface1::NodeDescriptor;
using interface1::SplitNodeDescriptor;
using interface1::TreeNodeVisitor;
} // namespace tree_utils
} // namespace algorithms
} // namespace daal

#endif
