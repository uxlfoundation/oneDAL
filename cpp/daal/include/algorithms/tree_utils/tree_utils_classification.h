/* file: tree_utils_classification.h */
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

#ifndef __TREE_UTILS_CLASSIFICATION__
#define __TREE_UTILS_CLASSIFICATION__

#include "algorithms/tree_utils/tree_utils.h"

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
namespace classification
{
/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__TREE_UTILS__CLASSIFICATION__LEAFNODEDESCRIPTOR"></a>
 * \brief %Struct containing description of leaf node in classification decision tree
 */
struct DAAL_EXPORT LeafNodeDescriptor : public NodeDescriptor
{
    size_t label;         /*!< Label to be predicted when reaching the leaf */
    const double * prob;  /*!< Probabilities estimation for the leaf */
    const double * cover; /*!< Cover (sum_hess) for the leaf */
};

typedef daal::algorithms::tree_utils::TreeNodeVisitor<LeafNodeDescriptor> TreeNodeVisitor;
typedef daal::algorithms::tree_utils::SplitNodeDescriptor SplitNodeDescriptor;

} // namespace interface2
using interface2::TreeNodeVisitor;
using interface2::SplitNodeDescriptor;
using interface2::LeafNodeDescriptor;
} // namespace classification
} // namespace tree_utils
} // namespace algorithms
} // namespace daal

#endif
