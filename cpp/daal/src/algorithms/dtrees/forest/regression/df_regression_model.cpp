/* file: df_regression_model.cpp */
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
//  Implementation of the class defining the decision forest model
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "src/services/serialization_utils.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/algorithms/dtrees/dtrees_model_impl_common.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
typedef decision_forest::regression::internal::ModelImpl::TreeType::NodeType::Leaf TLeaf;

namespace decision_forest
{
namespace regression
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_DECISION_FOREST_REGRESSION_MODEL_ID);

Model::Model() {}

} // namespace interface1

namespace internal
{
size_t ModelImpl::numberOfTrees() const
{
    return ImplType::size();
}

size_t ModelImpl::getNumberOfTrees() const
{
    return ImplType::size();
}

void ModelImpl::traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;
    const DecisionTreeTable & t    = *at(iTree);
    const DecisionTreeNode * aNode = (const DecisionTreeNode *)t.getArray();
    if (aNode)
    {
        auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, aNode, visitor);
        };

        auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool { return visitLeaf(iRowInTable, level, aNode, visitor); };

        traverseNodeDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;
    const DecisionTreeTable & t    = *at(iTree);
    const DecisionTreeNode * aNode = (const DecisionTreeNode *)t.getArray();
    NodeIdxArray aCur;  //nodes of current layer
    NodeIdxArray aNext; //nodes of next layer
    if (aNode)
    {
        aCur.push_back(0);

        auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, aNode, visitor);
        };

        auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool { return visitLeaf(iRowInTable, level, aNode, visitor); };

        traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;
    const DecisionTreeTable & t    = *at(iTree);
    const DecisionTreeNode * aNode = (const DecisionTreeNode *)t.getArray();
    const double * imp             = getImpVals(iTree);
    const int * nodeSamplesCount   = getNodeSampleCount(iTree);
    if (aNode)
    {
        tree_utils::SplitNodeDescriptor descSplit;
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
        };

        auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
        };

        traverseNodeDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

void ModelImpl::traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const
{
    if (iTree >= size()) return;
    const DecisionTreeTable & t    = *at(iTree);
    const DecisionTreeNode * aNode = (const DecisionTreeNode *)t.getArray();
    const double * imp             = getImpVals(iTree);
    const int * nodeSamplesCount   = getNodeSampleCount(iTree);
    NodeIdxArray aCur;  //nodes of current layer
    NodeIdxArray aNext; //nodes of next layer
    if (aNode)
    {
        tree_utils::SplitNodeDescriptor descSplit;
        tree_utils::regression::LeafNodeDescriptor descLeaf;

        auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
        };

        auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
            return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
        };

        aCur.push_back(0);
        traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
    }
}

services::Status ModelImpl::serializeImpl(data_management::InputDataArchive * arch)
{
    auto s = daal::algorithms::regression::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    s.add(ImplType::serialImpl<data_management::InputDataArchive, false>(arch));
    arch->set(daal::algorithms::regression::internal::ModelInternal::_nFeatures);

    return s;
}

services::Status ModelImpl::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    auto s = daal::algorithms::regression::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    s.add(ImplType::serialImpl<const data_management::OutputDataArchive, true>(arch));
    arch->set(RegressionImplType::_nFeatures);

    return s;
}

void ModelImpl::copyModelReg(const ModelImpl & other, size_t idx, size_t global_count)
{
    if (idx == 0)
    {
        resize(global_count); // sets _nTree = 0
        _nTree.set(global_count);
    }
    // copy data if source and target pointers are valid
    if (isValid() && other.isValid())
    {
        for (size_t i = 0; i < other._nTree.get(); i++)
        {
            auto probtbl                   = HomogenNumericTable<>::create(0, 0, NumericTable::doNotAllocate);
            (*_serializationData)[idx + i] = (*other._serializationData)[i];
            (*_impurityTables)[idx + i]    = (*other._impurityTables)[i];
            (*_nNodeSampleTables)[idx + i] = (*other._nNodeSampleTables)[i];
            (*_probTbl)[idx + i]           = probtbl;
        }
    }
}

bool ModelImpl::add(const TreeType & tree, size_t nClasses, size_t iTree)
{
    DAAL_CHECK_STATUS_VAR(!(size() >= _serializationData->size()));
    _nTree.inc();
    const size_t nNode = tree.getNumberOfNodes();

    auto pTbl           = new DecisionTreeTable(nNode);
    auto impTbl         = new HomogenNumericTable<double>(1, nNode, NumericTable::doAllocate);
    auto nodeSamplesTbl = new HomogenNumericTable<int>(1, nNode, NumericTable::doAllocate);
    auto probTbl        = new HomogenNumericTable<double>(0, 0, NumericTable::doAllocate);

    if (!pTbl || !impTbl || !nodeSamplesTbl || !probTbl)
    {
        delete pTbl;
        delete impTbl;
        delete nodeSamplesTbl;
        delete probTbl;
        return false;
    }

    tree.convertToTable(pTbl, impTbl, nodeSamplesTbl, probTbl, 0);

    (*_serializationData)[iTree].reset(pTbl);
    (*_impurityTables)[iTree].reset(impTbl);
    (*_nNodeSampleTables)[iTree].reset(nodeSamplesTbl);
    (*_probTbl)[iTree].reset(probTbl);

    return true;
}

} // namespace internal
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
