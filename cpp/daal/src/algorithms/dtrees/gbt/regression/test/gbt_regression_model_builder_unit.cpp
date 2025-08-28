/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/test/engine/common.hpp"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"

namespace daal::algorithms::gbt::internal
{
GbtDecisionTree prepareThreeNodeTree()
{
    /**
     * create a tree with 3 nodes, 1 root (split), 2 leaves
     *               ROOT        (level 1)
     *              /    \
     *             L      L      (level 2)
     */
    GbtDecisionTree tree = GbtDecisionTree(3, 2);

    ModelFPType * splitPoints       = tree.getSplitPoints();
    FeatureIndexType * splitIndices = tree.getFeatureIndexesForSplit();
    int * defaultLeft               = tree.getDefaultLeftForSplit();
    ModelFPType * coverValues       = tree.getNodeCoverValues();

    splitPoints[0]  = 1;
    splitIndices[0] = 0;
    defaultLeft[0]  = 1;
    coverValues[0]  = 1;

    splitPoints[1]  = 10;
    splitIndices[1] = 0;
    defaultLeft[1]  = 0;
    coverValues[1]  = 0.5;

    splitPoints[2]  = 11;
    splitIndices[2] = 0;
    defaultLeft[2]  = 0;
    coverValues[2]  = 0.5;

    return tree;
}

GbtDecisionTree prepareFiveNodeTree()
{
    /** create a tree with 5 nodes
     *               ROOT (1)        (level 1)
     *               /    \
     *             L (2)  S (3)      (level 2)
     *                    / \
     *                L (6) L (7)    (level 3)
     * (note: on level 3, nodes 4 and 5 do not exist and will be created as "dummy leaf")
     */
    GbtDecisionTree tree = GbtDecisionTree(5, 3);

    ModelFPType * splitPoints       = tree.getSplitPoints();
    FeatureIndexType * splitIndices = tree.getFeatureIndexesForSplit();
    int * defaultLeft               = tree.getDefaultLeftForSplit();
    ModelFPType * coverValues       = tree.getNodeCoverValues();

    // node idx 1
    splitPoints[0]  = 1;
    splitIndices[0] = 0;
    defaultLeft[0]  = 1;
    coverValues[0]  = 10;

    // node idx 2
    // the node with dummy leaf children
    splitPoints[1]  = 10;
    splitIndices[1] = 20;
    defaultLeft[1]  = 0;
    coverValues[1]  = 4;

    // node idx 3
    splitPoints[2]  = 11;
    splitIndices[2] = 0;
    defaultLeft[2]  = 0;
    coverValues[2]  = 6;

    // node idx 4 (dummy leaf)
    // split point and value equal to parent node
    splitPoints[3]  = splitPoints[1];
    splitIndices[3] = splitIndices[1];
    defaultLeft[3]  = 0;
    coverValues[3]  = 0;

    // node idx 5 (dummy leaf)
    // split point and value equal to parent node
    splitPoints[4]  = splitPoints[1];
    splitIndices[4] = splitIndices[1];
    defaultLeft[4]  = 0;
    coverValues[4]  = 0;

    // node idx 6
    splitPoints[5]  = 12;
    splitIndices[5] = 22;
    defaultLeft[5]  = 0;
    coverValues[5]  = 4;

    // node idx 7
    splitPoints[6]  = 13;
    splitIndices[6] = 23;
    defaultLeft[6]  = 0;
    coverValues[6]  = 2;

    return tree;
}

TEST("nodeIsLeafThreeNodes", "[unit]")
{
    GbtDecisionTree tree = prepareThreeNodeTree();

    REQUIRE(!ModelImpl::nodeIsLeaf(1, tree, 1));
    REQUIRE(ModelImpl::nodeIsLeaf(2, tree, 2));
    REQUIRE(ModelImpl::nodeIsLeaf(3, tree, 2));
}

TEST("nodeIsDummyLeafFiveNodes", "[unit]")
{
    GbtDecisionTree tree = prepareFiveNodeTree();
    // TODO update test
    REQUIRE(!ModelImpl::nodeIsDummyLeaf(1, tree));
    REQUIRE(!ModelImpl::nodeIsDummyLeaf(2, tree));
    REQUIRE(!ModelImpl::nodeIsDummyLeaf(3, tree));
    REQUIRE(ModelImpl::nodeIsDummyLeaf(4, tree));
    REQUIRE(ModelImpl::nodeIsDummyLeaf(5, tree));
    REQUIRE(!ModelImpl::nodeIsDummyLeaf(6, tree));
    REQUIRE(!ModelImpl::nodeIsDummyLeaf(7, tree));
}

TEST("nodeIsLeafFiveNodes", "[unit]")
{
    GbtDecisionTree tree = prepareFiveNodeTree();

    REQUIRE(!ModelImpl::nodeIsLeaf(1, tree, 1));
    REQUIRE(ModelImpl::nodeIsLeaf(2, tree, 2));
    REQUIRE(!ModelImpl::nodeIsLeaf(3, tree, 2));
    REQUIRE(ModelImpl::nodeIsLeaf(6, tree, 3));
    REQUIRE(ModelImpl::nodeIsLeaf(7, tree, 3));
}
} // namespace daal::algorithms::gbt::internal
