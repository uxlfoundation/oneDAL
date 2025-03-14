/* file: tree_traverse.h */
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

#ifndef __TREE_CLASSIFIER_TRAVERSE__
#define __TREE_CLASSIFIER_TRAVERSE__

namespace daal
{
namespace algorithms
{
/**
 * @defgroup classification Classification
 * \brief Contains classes for work with the classification algorithms
 * @ingroup training_and_prediction
 */
namespace classifier
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup trees_classification
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TREENODEVISITOR"></a>
 * \brief %Interface of abstract visitor used in tree traversal methods. \DAAL_DEPRECATED
 *
 * \par References
 *      - \ref decision_forest::classification::interface1::Model "Model" class
 *      - \ref decision_tree::classification::interface1::Model "Model" class
 */
class DAAL_EXPORT TreeNodeVisitor
{
public:
    virtual ~TreeNodeVisitor() {}
    /**
    *  This method is called by traversal method when a leaf node is visited.
    *  \param[in] level    Level in the tree where this node is located
    *  \param[in] response The value of response given by that node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onLeafNode(size_t level, size_t response) = 0;
    /**
    *  This method is called by traversal method when a split node is visited.
    *  \param[in] level        Index of the feature used in a split criteria
    *  \param[in] featureIndex Index of the feature used as a split criteria in this node
    *  \param[in] featureValue Feature value used as a split criteria in this node
    *  \return This method should return false to cancel further search and true otherwise
    */
    virtual bool onSplitNode(size_t level, size_t featureIndex, double featureValue) = 0;
};

} // namespace interface1

using interface1::TreeNodeVisitor;

/** @} */
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
