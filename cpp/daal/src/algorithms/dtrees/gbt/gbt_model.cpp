/* file: gbt_model.cpp */
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace internal
{
typedef gbt::internal::ModelImpl::TreeType::NodeType::Leaf TLeaf;

int GbtDecisionTree::serializationTag()
{
    return _desc.tag();
}
int GbtDecisionTree::getSerializationTag() const
{
    return _desc.tag();
}
static data_management::SerializationIface * creatorGbtDecisionTree()
{
    return new GbtDecisionTree();
}
data_management::SerializationDesc GbtDecisionTree::_desc(creatorGbtDecisionTree, SERIALIZATION_GBT_DECISION_TREE_ID);

size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;

    const GbtDecisionTree & gbtTree = *at(iTree);

    const ModelFPType * splitPoints        = gbtTree.getSplitPoints();
    const FeatureIndexType * splitFeatures = gbtTree.getFeatureIndexesForSplit();

    auto onSplitNodeFunc = [&splitPoints, &splitFeatures, &visitor](size_t iRowInTable, size_t level) -> bool {
        return visitor.onSplitNode(level, splitFeatures[iRowInTable], splitPoints[iRowInTable]);
    };

    auto onLeafNodeFunc = [&splitPoints, &visitor](size_t iRowInTable, size_t level) -> bool {
        return visitor.onLeafNode(level, splitPoints[iRowInTable]);
    };

    traverseGbtDF(0, 0, gbtTree, onSplitNodeFunc, onLeafNodeFunc);
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;

    const GbtDecisionTree & gbtTree = *at(iTree);

    const ModelFPType * splitPoints        = gbtTree.getSplitPoints();
    const FeatureIndexType * splitFeatures = gbtTree.getFeatureIndexesForSplit();

    auto onSplitNodeFunc = [&splitFeatures, &splitPoints, &visitor](size_t iRowInTable, size_t level) -> bool {
        return visitor.onSplitNode(level, splitFeatures[iRowInTable], splitPoints[iRowInTable]);
    };

    auto onLeafNodeFunc = [&splitPoints, &visitor](size_t iRowInTable, size_t level) -> bool {
        return visitor.onLeafNode(level, splitPoints[iRowInTable]);
    };

    NodeIdxArray aCur;  //nodes of current layer
    NodeIdxArray aNext; //nodes of next layer

    aCur.push_back(0);

    traverseGbtBF(0, aCur, aNext, gbtTree, onSplitNodeFunc, onLeafNodeFunc);
}

void ModelImpl::traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;

    const GbtDecisionTree & gbtTree = *at(iTree);

    const ModelFPType * splitPoints        = gbtTree.getSplitPoints();
    const FeatureIndexType * splitFeatures = gbtTree.getFeatureIndexesForSplit();
    const int * nodeSamplesCount           = getNodeSampleCount(iTree);
    const double * imp                     = getImpVals(iTree);

    auto onSplitNodeFunc = [&splitFeatures, &splitPoints, &nodeSamplesCount, &imp, &visitor](size_t iRowInTable, size_t level) -> bool {
        tree_utils::SplitNodeDescriptor descSplit;

        descSplit.impurity         = imp != nullptr ? imp[iRowInTable] : 0;
        descSplit.nNodeSampleCount = nodeSamplesCount != nullptr ? (size_t)(nodeSamplesCount[iRowInTable]) : 0;
        descSplit.featureIndex     = splitFeatures[iRowInTable];
        descSplit.featureValue     = splitPoints[iRowInTable];
        descSplit.level            = level;
        return visitor.onSplitNode(descSplit);
    };

    auto onLeafNodeFunc = [&splitPoints, &nodeSamplesCount, &imp, &visitor](size_t iRowInTable, size_t level) -> bool {
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        descLeaf.impurity         = imp != nullptr ? imp[iRowInTable] : 0;
        descLeaf.nNodeSampleCount = nodeSamplesCount != nullptr ? (size_t)(nodeSamplesCount[iRowInTable]) : 0;
        descLeaf.level            = level;
        descLeaf.response         = splitPoints[iRowInTable];
        return visitor.onLeafNode(descLeaf);
    };

    NodeIdxArray aCur;  //nodes of current layer
    NodeIdxArray aNext; //nodes of next layer

    aCur.push_back(0);

    traverseGbtBF(0, aCur, aNext, gbtTree, onSplitNodeFunc, onLeafNodeFunc);
}

void ModelImpl::traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;

    const GbtDecisionTree & gbtTree = *at(iTree);

    const ModelFPType * splitPoints        = gbtTree.getSplitPoints();
    const FeatureIndexType * splitFeatures = gbtTree.getFeatureIndexesForSplit();
    const int * nodeSamplesCount           = getNodeSampleCount(iTree);
    const double * imp                     = getImpVals(iTree);

    auto onSplitNodeFunc = [&splitFeatures, &splitPoints, &nodeSamplesCount, &imp, &visitor](size_t iRowInTable, size_t level) -> bool {
        tree_utils::SplitNodeDescriptor descSplit;

        descSplit.impurity         = imp != nullptr ? imp[iRowInTable] : 0;
        descSplit.nNodeSampleCount = nodeSamplesCount != nullptr ? (size_t)(nodeSamplesCount[iRowInTable]) : 0;
        descSplit.featureIndex     = splitFeatures[iRowInTable];
        descSplit.featureValue     = splitPoints[iRowInTable];
        descSplit.level            = level;
        return visitor.onSplitNode(descSplit);
    };

    auto onLeafNodeFunc = [&splitPoints, &nodeSamplesCount, &imp, &visitor](size_t iRowInTable, size_t level) -> bool {
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        descLeaf.impurity         = imp != nullptr ? imp[iRowInTable] : 0;
        descLeaf.nNodeSampleCount = nodeSamplesCount != nullptr ? (size_t)(nodeSamplesCount[iRowInTable]) : 0;
        descLeaf.level            = level;
        descLeaf.response         = splitPoints[iRowInTable];
        return visitor.onLeafNode(descLeaf);
    };

    traverseGbtDF(0, 0, gbtTree, onSplitNodeFunc, onLeafNodeFunc);
}

services::Status ModelImpl::treeToTable(TreeType & t, gbt::internal::GbtDecisionTree ** pTbl, HomogenNumericTable<double> ** pTblImp,
                                        HomogenNumericTable<int> ** pTblSmplCnt, size_t nFeat)
{
    return t.convertGbtTreeToTable(pTbl, pTblImp, pTblSmplCnt, nFeat);
}

void ModelImpl::add(gbt::internal::GbtDecisionTree * pTbl, HomogenNumericTable<double> * pTblImp, HomogenNumericTable<int> * pTblSmplCnt)
{
    DAAL_ASSERT(pTbl);
    DAAL_ASSERT(pTblImp);
    DAAL_ASSERT(pTblSmplCnt);

    _nTree.inc();

    _serializationData->push_back(SerializationIfacePtr(pTbl));
    _impurityTables->push_back(SerializationIfacePtr(pTblImp));
    _nNodeSampleTables->push_back(SerializationIfacePtr(pTblSmplCnt));
}

ModelImpl::~ModelImpl()
{
    destroy();
}

size_t ModelImpl::size() const
{
    return super::size();
}

bool ModelImpl::reserve(const size_t nTrees)
{
    return super::reserve(nTrees);
}

bool ModelImpl::resize(const size_t nTrees)
{
    return super::resize(nTrees);
}

void ModelImpl::clear()
{
    super::clear();
}

void ModelImpl::destroy()
{
    super::destroy();
}

bool ModelImpl::nodeIsLeaf(size_t idx, const GbtDecisionTree & gbtTree, const size_t lvl)
{
    const SplitLeftIdPair * splitAndLeftIds = gbtTree.getSplitsAndLeftIds();
    size_t leftId                           = splitAndLeftIds[idx - 1].leftId;
    if (leftId == idx)
    {
        return true;
    }

    const FeatureIndexType * splitFeatures = gbtTree.getFeatureIndexesForSplit();
    return (splitAndLeftIds[leftId - 1].splitPoint == splitAndLeftIds[idx - 1].splitPoint && splitFeatures[leftId - 1] == splitFeatures[idx - 1]);
}

void ModelImpl::decisionTreeToGbtTree(const DecisionTreeTable & tree, GbtDecisionTree & newTree)
{
    const size_t nSourceNodes = tree.getNumberOfRows();
    const size_t nLvls        = newTree.getMaxLvl();
    const size_t nDenseLayers = newTree.getNumDenseLayers();

    using NodeType = const dtrees::internal::DecisionTreeNode *;
    services::Collection<NodeType> sonsArr(newTree.getNumberOfNodes() + 1);
    services::Collection<NodeType> parentsArr(newTree.getNumberOfNodes() + 1);
    NodeType arr = (const NodeType)tree.getArray();

    NodeType * sons    = sonsArr.data();
    NodeType * parents = parentsArr.data();

    ModelFPType * const splitPoints         = newTree.getSplitPoints();
    size_t * const leftChildIndexes         = newTree.getLeftChildIndexes();
    SplitLeftIdPair * const SplitAndLeftIds = newTree.getSplitsAndLeftIds();
    FeatureIndexType * const featureIndexes = newTree.getFeatureIndexesForSplit();
    ModelFPType * const nodeCoverValues     = newTree.getNodeCoverValues();
    int * const defaultLeft                 = newTree.getDefaultLeftForSplit();

    for (size_t i = 0; i < nSourceNodes; ++i)
    {
        sons[i]    = nullptr;
        parents[i] = nullptr;
    }

    size_t nParents   = 1;
    parents[0]        = arr;
    size_t idxInTable = 0;
    size_t idxChild   = 2;

    for (size_t lvl = 0; lvl < nLvls + 1; ++lvl)
    {
        size_t nSons = 0;
        for (size_t iParent = 0; iParent < nParents; ++iParent)
        {
            const dtrees::internal::DecisionTreeNode * p = parents[iParent];

            if (p->isSplit())
            {
                sons[nSons++]               = arr + p->leftIndexOrClass;
                sons[nSons++]               = arr + p->leftIndexOrClass + 1;
                featureIndexes[idxInTable]  = p->featureIndex;
                nodeCoverValues[idxInTable] = p->cover;
                defaultLeft[idxInTable]     = p->defaultLeft;
                DAAL_ASSERT(featureIndexes[idxInTable] >= 0);
                splitPoints[idxInTable] = p->featureValueOrResponse;

                leftChildIndexes[idxInTable]           = idxChild;
                SplitAndLeftIds[idxInTable].leftId     = idxChild;
                SplitAndLeftIds[idxInTable].splitPoint = p->featureValueOrResponse;
                idxChild += 2;
            }
            else
            {
                if (lvl < std::min(nDenseLayers, nLvls))
                {
                    sons[nSons++] = p;
                    sons[nSons++] = p;

                    featureIndexes[idxInTable]  = 0;
                    nodeCoverValues[idxInTable] = p->cover;
                    defaultLeft[idxInTable]     = 0;
                    splitPoints[idxInTable]     = p->featureValueOrResponse;

                    leftChildIndexes[idxInTable]           = idxChild;
                    SplitAndLeftIds[idxInTable].leftId     = idxChild;
                    SplitAndLeftIds[idxInTable].splitPoint = p->featureValueOrResponse;
                    featureIndexes[idxInTable]             = 0;

                    idxChild += 2;
                }
                else
                {
                    featureIndexes[idxInTable]  = 0;
                    nodeCoverValues[idxInTable] = p->cover;
                    defaultLeft[idxInTable]     = 0;
                    splitPoints[idxInTable]     = p->featureValueOrResponse;

                    leftChildIndexes[idxInTable]           = idxInTable + 1;
                    SplitAndLeftIds[idxInTable].leftId     = idxInTable + 1;
                    SplitAndLeftIds[idxInTable].splitPoint = p->featureValueOrResponse;
                }
            }
            idxInTable++;
        }
        swap(parents, sons);
        nParents = nSons;
    }
}

services::Status ModelImpl::convertDecisionTreesToGbtTrees(data_management::DataCollectionPtr & serializationData)
{
    services::Status s;
    const size_t size                          = serializationData->size();
    data_management::DataCollection * newTrees = new data_management::DataCollection();
    DAAL_CHECK_MALLOC(newTrees)
    for (size_t i = 0; i < size; ++i)
    {
        const DecisionTreeTable & tree = *(DecisionTreeTable *)(*(serializationData))[i].get();
        GbtDecisionTree * newTree      = allocateGbtTree(tree);
        decisionTreeToGbtTree(tree, *newTree);
        newTrees->push_back(SerializationIfacePtr(newTree));
    }
    serializationData.reset(newTrees);
    return s;
}

void ModelImpl::getMaxLvl(const dtrees::internal::DecisionTreeNode * const arr, const size_t idx, size_t & maxLvl, size_t curLvl)
{
    curLvl++;

    if (arr[idx].isSplit())
    {
        getMaxLvl(arr, arr[idx].leftIndexOrClass, maxLvl, curLvl);
        getMaxLvl(arr, arr[idx].leftIndexOrClass + 1, maxLvl, curLvl);
    }
    else
    {
        if (maxLvl < curLvl) maxLvl = curLvl;
    }
}

void ModelImpl::getMaxLvLAndNumNodes(const dtrees::internal::DecisionTreeNode * const arr, const size_t idx, size_t & maxLvl, size_t & numNodes,
                                     const size_t numDenseLayers, size_t curLvl)
{
    if (curLvl > numDenseLayers)
    {
        numNodes++;
    }
    if (arr[idx].isSplit())
    {
        getMaxLvLAndNumNodes(arr, arr[idx].leftIndexOrClass, maxLvl, numNodes, numDenseLayers, curLvl + 1);
        getMaxLvLAndNumNodes(arr, arr[idx].leftIndexOrClass + 1, maxLvl, numNodes, numDenseLayers, curLvl + 1);
    }
    else
    {
        if (maxLvl < curLvl) maxLvl = curLvl;
    }
}

const GbtDecisionTree * ModelImpl::at(const size_t idx) const
{
    auto * const rawTree = (*super::_serializationData)[idx].get();
    return static_cast<const GbtDecisionTree *>(rawTree);
}

} // namespace internal
} // namespace gbt
} // namespace algorithms
} // namespace daal
