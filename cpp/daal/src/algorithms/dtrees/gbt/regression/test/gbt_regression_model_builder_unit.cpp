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
#include "algorithms/gradient_boosted_trees/gbt_regression_model_builder.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include <vector>
#include <random>

namespace daal::algorithms::gbt::internal
{
using namespace daal::algorithms::gbt::regression;
using namespace daal::algorithms::gbt::regression::interface1;
using ModelImplType = daal::algorithms::gbt::regression::internal::ModelImpl;

TEST("DecisionTreeToGbtModelConversion_SimpleTree", "[unit]")
{
    /**
     * Build a simple decision tree with 3 nodes using ModelBuilder API:
     *               ROOT (split)     (level 0)
     *              /    \
     *          LEAF1    LEAF2        (level 1)
     */
    size_t nFeatures = 5;
    size_t nTrees    = 1;

    // Create model builder
    ModelBuilder builder(nFeatures, nTrees);

    // Create a tree with 3 nodes
    auto treeId = builder.createTree(3);

    // Add root split node
    auto rootNode = builder.addSplitNode(treeId,
                                         ModelBuilder::noParent, // root node
                                         0,                      // position (not used for root)
                                         0,                      // feature index
                                         5.0,                    // split value
                                         1,                      // default left
                                         10.0                    // cover
    );

    // Add left child leaf
    builder.addLeafNode(treeId, rootNode,
                        0,   // left child position
                        1.5, // response value
                        6.0  // cover
    );

    // Add right child leaf
    builder.addLeafNode(treeId, rootNode,
                        1,   // right child position
                        2.5, // response value
                        4.0  // cover
    );

    // Convert to GBT model using getModel interface
    auto gbtModel = builder.getModel();
    REQUIRE(gbtModel.get() != nullptr);

    // Get model implementation to access internal GBT structure
    auto & modelImpl = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,
                                                                       daal::algorithms::gbt::regression::interface1::ModelPtr>(gbtModel);

    // Verify tree count
    REQUIRE(modelImpl.numberOfTrees() == 1);

    // Get the GBT tree
    auto gbtTree = modelImpl.at(0);
    REQUIRE(gbtTree != nullptr);

    // Verify tree structure
    const size_t expectedNodeCount = 3;
    REQUIRE(gbtTree->getNumberOfNodes() == expectedNodeCount);

    // Verify split points
    const auto * splitPoints = gbtTree->getSplitPoints();
    REQUIRE(splitPoints != nullptr);
    REQUIRE(splitPoints[0] == 5.0); // Root split value

    // Verify feature indexes
    const auto * featureIndexes = gbtTree->getFeatureIndexesForSplit();
    REQUIRE(featureIndexes != nullptr);
    REQUIRE(featureIndexes[0] == 0); // Root feature index

    // Verify node cover values
    const auto * coverValues = gbtTree->getNodeCoverValues();
    REQUIRE(coverValues != nullptr);
    REQUIRE(coverValues[0] == 10.0); // Root cover

    // Verify left child indexes
    const auto * leftChildIndexes = gbtTree->getLeftChildIndexes();
    REQUIRE(leftChildIndexes != nullptr);
    // Root node should have left child at index 2 (1-based indexing, left child at position 2)
    REQUIRE(leftChildIndexes[0] == 2);

    // Left and right leaves should not have children (leftId should point to themselves)
    REQUIRE(leftChildIndexes[1] == 2);
    REQUIRE(leftChildIndexes[2] == 3);
}

TEST("DecisionTreeToGbtModelConversion_MultiLevelTree", "[unit]")
{
    /**
     * Build a multi-level decision tree:
     *               ROOT (split)         (level 0)
     *              /    \
     *        SPLIT1    LEAF2             (level 1)
     *        /    \
     *    LEAF3   LEAF4                   (level 2)
     */
    size_t nFeatures = 3;
    size_t nTrees    = 1;

    ModelBuilder builder(nFeatures, nTrees);

    // Create a tree with 5 nodes
    auto treeId = builder.createTree(5);

    // Add root split node
    auto rootNode = builder.addSplitNode(treeId, ModelBuilder::noParent, 0,
                                         0,    // feature index
                                         10.0, // split value
                                         1,    // default left
                                         20.0  // cover
    );

    // Add left child split node
    auto leftSplit = builder.addSplitNode(treeId, rootNode,
                                          0,   // left position
                                          1,   // feature index
                                          7.0, // split value
                                          0,   // default left
                                          12.0 // cover
    );

    // Add right child leaf (of root)
    builder.addLeafNode(treeId, rootNode,
                        1,   // right position
                        3.0, // response
                        8.0  // cover
    );

    // Add left child leaf (of leftSplit)
    builder.addLeafNode(treeId, leftSplit,
                        0,   // left position
                        1.0, // response
                        5.0  // cover
    );

    // Add right child leaf (of leftSplit)
    builder.addLeafNode(treeId, leftSplit,
                        1,   // right position
                        2.0, // response
                        7.0  // cover
    );

    // Convert using getModel
    auto gbtModel = builder.getModel();
    REQUIRE(gbtModel.get() != nullptr);

    auto & modelImpl = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,
                                                                       daal::algorithms::gbt::regression::interface1::ModelPtr>(gbtModel);

    // Verify tree count
    REQUIRE(modelImpl.numberOfTrees() == 1);

    // Get the GBT tree
    auto gbtTree = modelImpl.at(0);
    REQUIRE(gbtTree != nullptr);

    // Verify node count - should be 7 nodes, all layers are stored in dense format
    size_t nodeCount = gbtTree->getNumberOfNodes();
    REQUIRE(nodeCount == 7); // May have more nodes due to layout structure

    // Verify split points array
    const auto * splitPoints = gbtTree->getSplitPoints();
    REQUIRE(splitPoints != nullptr);
    REQUIRE(splitPoints[0] == 10.0); // ROOT
    REQUIRE(splitPoints[1] == 7.0);  // SPLIT1
    REQUIRE(splitPoints[2] == 3.0);  // LEAF2 response value
    REQUIRE(splitPoints[3] == 1.0);  // LEAF3 response value
    REQUIRE(splitPoints[4] == 2.0);  // LEAF4 response value
    REQUIRE(splitPoints[5] == 3.0);  // dummy leaf response value (copy of LEAF2)
    REQUIRE(splitPoints[6] == 3.0);  // dummy leaf response value (copy of LEAF2)

    // Verify left child indexes structure
    const auto * leftChildIndexes = gbtTree->getLeftChildIndexes();
    REQUIRE(leftChildIndexes != nullptr);

    // Split node leftId should be (nodeId + 1) * 2, considering 1-based indexing
    REQUIRE(leftChildIndexes[0] == 2);
    REQUIRE(leftChildIndexes[1] == 4);
    REQUIRE(leftChildIndexes[2] == 6);

    // Leaves leftId should point to themselves
    REQUIRE(leftChildIndexes[3] == 4);
    REQUIRE(leftChildIndexes[4] == 5);
    REQUIRE(leftChildIndexes[5] == 6);
    REQUIRE(leftChildIndexes[6] == 7);

    // Verify feature indexes
    const auto * featureIndexes = gbtTree->getFeatureIndexesForSplit();
    REQUIRE(featureIndexes != nullptr);
    REQUIRE(featureIndexes[0] == 0); // Root feature

    // Verify cover values are set correctly
    const auto * coverValues = gbtTree->getNodeCoverValues();
    REQUIRE(coverValues != nullptr);
    REQUIRE(coverValues[0] == 20.0); // Root cover
}

TEST("DecisionTreeToGbtModelConversion_SparseRandomTree", "[unit]")
{
    /**
     * Build a sparse random tree with max level 12-13 and approximately 100 nodes
     * using a fixed random seed for reproducibility.
     *
     * Algorithm:
     * 1. Create root as split node, add leaf info to current_leaves list (don't create leaf yet)
     * 2. While current_leaves has nodes and nodesCreated < targetNodeCount:
     *    - Randomly select a leaf info from current_leaves
     *    - Create split node with this leaf info
     *    - Add two new leaf infos as children to current_leaves (don't create leaves yet)
     * 3. Create all remaining leaf nodes from current_leaves array
     */
    const size_t nFeatures        = 10;
    const size_t nTrees           = 1;
    const size_t maxLevel         = 13;
    const size_t targetNodeCount  = 256;
    const unsigned int randomSeed = 42;
    const int defaultLeft         = 1;   // Same for all tree
    const double coverValue       = 1.0; // Constant

    // Create model builder
    ModelBuilder builder(nFeatures, nTrees);
    auto treeId = builder.createTree(targetNodeCount);

    // Use deterministic random number generator
    std::mt19937 rng(randomSeed);
    std::uniform_int_distribution<size_t> featureDist(0, nFeatures - 1);
    std::uniform_real_distribution<double> splitValueDist(0.0, 100.0);

    // Store leaf node information (don't create leaves yet)
    struct LeafInfo
    {
        ModelBuilder::NodeId parentId;
        size_t position;
        size_t level;
    };

    // Initialize current_leaves list with info for two children of root
    std::vector<LeafInfo> currentLeaves;
    auto rootNode = builder.addSplitNode(treeId, ModelBuilder::noParent, 0, featureDist(rng), splitValueDist(rng), defaultLeft, coverValue);

    // Add two initial leaf infos (don't create leaves yet)
    currentLeaves.push_back({ rootNode, 0, 1 });
    currentLeaves.push_back({ rootNode, 1, 1 });

    size_t nodesCreated = 3;

    // Generate tree structure by converting random leaves to split nodes
    while (!currentLeaves.empty() && nodesCreated + 2 <= targetNodeCount)
    {
        // Update distribution range based on current leaves count
        std::uniform_int_distribution<size_t> leafIndexDist(0, currentLeaves.size() - 1);

        // Randomly select a leaf info from current_leaves
        size_t selectedIndex  = leafIndexDist(rng);
        LeafInfo selectedLeaf = currentLeaves[selectedIndex];

        // Check if this leaf can have children (max level constraint)
        if (selectedLeaf.level >= maxLevel)
        {
            // We cannot exceed maximum depth
            continue;
        }

        // Convert the selected leaf to a split node
        auto splitNode = builder.addSplitNode(treeId, selectedLeaf.parentId, selectedLeaf.position, featureDist(rng), splitValueDist(rng),
                                              defaultLeft, coverValue);

        // Add two new leaf infos as children (don't create leaves yet)
        currentLeaves.push_back({ splitNode, 0, selectedLeaf.level + 1 });
        currentLeaves.push_back({ splitNode, 1, selectedLeaf.level + 1 });

        nodesCreated += 2;

        // Remove the converted leaf from the list
        currentLeaves.erase(currentLeaves.begin() + selectedIndex);
    }

    // Now create all remaining leaf nodes from current_leaves array
    for (const auto & leafInfo : currentLeaves)
    {
        builder.addLeafNode(treeId, leafInfo.parentId, leafInfo.position,
                            1.0, // response value
                            coverValue);
    }

    // Convert to GBT model
    auto gbtModel = builder.getModel();
    REQUIRE(gbtModel.get() != nullptr);

    auto & modelImpl = daal::algorithms::dtrees::internal::getModelRef<daal::algorithms::gbt::regression::internal::ModelImpl,
                                                                       daal::algorithms::gbt::regression::interface1::ModelPtr>(gbtModel);

    // Verify tree count
    REQUIRE(modelImpl.numberOfTrees() == 1);

    // Get the GBT tree
    auto gbtTree = modelImpl.at(0);
    REQUIRE(gbtTree != nullptr);

    // Verify node count is reasonable
    size_t nodeCount         = gbtTree->getNumberOfNodes();
    size_t nLevels           = gbtTree->getMaxLvl();
    size_t nActualDepthLevel = gbtTree->getNumDenseLayers();

    // Check that default number of dense layers is used
    REQUIRE(nActualDepthLevel == defaultNumDenseLayers);

    // Check that maximum depth is reached (to ensure test is sufficient)
    REQUIRE(nLevels == maxLevel);

    // Verify raw data is accessible
    const auto * splitPoints      = gbtTree->getSplitPoints();
    const auto * featureIndexes   = gbtTree->getFeatureIndexesForSplit();
    const auto * leftChildIndexes = gbtTree->getLeftChildIndexes();
    const auto * coverValues      = gbtTree->getNodeCoverValues();

    REQUIRE(splitPoints != nullptr);
    REQUIRE(featureIndexes != nullptr);
    REQUIRE(leftChildIndexes != nullptr);
    REQUIRE(coverValues != nullptr);

    // Analyze the tree structure
    // Calculate the number of leaves, dummy leaves and split nodes
    // Calculate the number of nodes at each layer

    std::vector<size_t> nodeLevels(nodeCount, 0);
    std::vector<size_t> nodesPerLevel(maxLevel + 1, 0);

    nodeLevels[0]    = 0;
    int nLeaves      = 0;
    int nDummyLeaves = 0;

    for (size_t i = 0; i < nodeCount; ++i)
    {
        REQUIRE(featureIndexes[i] < nFeatures);
        REQUIRE(coverValues[i] == 1.0);

        uint32_t leftId = leftChildIndexes[i];

        nodesPerLevel[nodeLevels[i]]++;
        int isLeaf = modelImpl.nodeIsLeaf(i + 1, *gbtTree);
        nLeaves += isLeaf;

        // Either leftId points to current node or to some other node
        REQUIRE((leftId - 1 == i || leftId < nodeCount));
        if (nodeLevels[i] < defaultNumDenseLayers)
        {
            // If node is on dense level it should have both children (possibly dummy nodes) at specified index
            REQUIRE(leftId == (i + 1) * 2);
        }
        else
        {
            if (isLeaf)
            {
                // If node is on deep level and is a leaf it shouldn't have children
                REQUIRE(leftId - 1 == i);
            }
            else
            {
                REQUIRE(leftId - 1 > i);
            }
        }

        if (isLeaf && leftId - 1 != i)
        {
            // Current node is a lead and its children are dummy leaves
            nDummyLeaves += 2;
        }

        if (leftId - 1 != i && leftId < nodeCount)
        {
            nodeLevels[leftId - 1] = nodeLevels[i] + 1;
            nodeLevels[leftId]     = nodeLevels[i] + 1;
        }
    }

    size_t deepNodes = 0;
    for (size_t i = 0; i < maxLevel + 1; ++i)
    {
        if (i <= defaultNumDenseLayers)
        {
            REQUIRE(nodesPerLevel[i] == (1u << i));
        }
        else
        {
            // To ensure we have nodes on all levels and test is sufficient
            REQUIRE(nodesPerLevel[i] > 0);
            deepNodes += nodesPerLevel[i];
        }
    }

    size_t nRealLeaves = nLeaves - nDummyLeaves;
    size_t nSplitNodes = nodeCount - nLeaves;

    // Verify total node count matches expected structure
    // First defaultNumDenseLayers levels are full binary tree, the rest structure is sparse
    REQUIRE(nodeCount == (1u << (defaultNumDenseLayers + 1)) - 1 + deepNodes);

    // Number of meaningful nodes correspond to the number of nodes in original tree
    REQUIRE(nodesCreated == nRealLeaves + nSplitNodes);

    // Number of leaves is number of split nodes + 1 is a binary tree
    REQUIRE(nRealLeaves == nSplitNodes + 1);
}

} // namespace daal::algorithms::gbt::internal
