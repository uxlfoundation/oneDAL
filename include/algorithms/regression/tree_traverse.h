/* file: tree_traverse.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __TREE_REGRESSION_TRAVERSE__
#define __TREE_REGRESSION_TRAVERSE__

namespace daal
{
namespace algorithms
{
/**
 * @defgroup trees_regression Tree regression
 * \copydoc daal::algorithms::trees::regression
 * @ingroup regression
 */

/**
 * \brief Contains classes for tree regression algorithms
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup trees_regression
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TREENODEVISITOR"></a>
 * \brief %Interface of abstract visitor used in tree traversal methods. \DAAL_DEPRECATED
 *
 * \par References
 *      - \ref decision_forest::regression::interface1::Model "Model" class
 *      - \ref decision_tree::regression::interface1::Model "Model" class
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
    virtual bool onLeafNode(size_t level, double response) = 0;
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
} // namespace regression
} // namespace algorithms
} // namespace daal

#endif
