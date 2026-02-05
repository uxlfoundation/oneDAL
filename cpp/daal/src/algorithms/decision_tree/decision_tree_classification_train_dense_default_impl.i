/* file: decision_tree_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for Decision tree dense default method.
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DECISION_TREE_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/services/service_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/decision_tree/decision_tree_classification_model_impl.h"
#include "src/algorithms/decision_tree/decision_tree_classification_train_kernel.h"
#include "src/algorithms/decision_tree/decision_tree_train_impl.i"
#include "src/algorithms/decision_tree/decision_tree_classification_split_criterion.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace decision_tree::internal;

template <CpuType cpu>
class REPPruningData : public PruningData<cpu, int>
{
    typedef PruningData<cpu, int> BaseType;

public:
    REPPruningData(size_t size, size_t numberOfClasses)
        : BaseType(size), _numberOfClasses(numberOfClasses), _counters(daal_alloc<size_t>(max<cpu, size_t>(1, size * numberOfClasses)))
    {
        resetCounters();
    }

    ~REPPruningData() override
    {
        daal_free(_counters);
        _counters = nullptr;
    }

    void update(size_t index, size_t classIndex)
    {
        DAAL_ASSERT(index < size());
        DAAL_ASSERT(classIndex < _numberOfClasses);
        ++_counters[index * _numberOfClasses + classIndex];
    }

    typedef size_t ErrorType;

    ErrorType error(size_t index, size_t classIndex) const
    {
        DAAL_ASSERT(index < size());
        DAAL_ASSERT(classIndex < _numberOfClasses);

        const size_t * const c = &_counters[index * _numberOfClasses];

        ErrorType err = 0;
        for (size_t i = 0; i < _numberOfClasses; ++i)
        {
            err += c[i];
        }

        return err - c[classIndex];
    }

    ErrorType error(size_t index) const
    {
        DAAL_ASSERT(index < size());

        const size_t * const c = &_counters[index * _numberOfClasses];

        ErrorType err = 0, max = 0;
        for (size_t i = 0; i < _numberOfClasses; ++i)
        {
            err += c[i];

            if (c[i] > max)
            {
                max = c[i];
            }
        }

        return err - max;
    }

    void prune(size_t index) { BaseType::prune(index, majorityClass(index)); }

    void putProbabilities(size_t index, double * probs, size_t numProbs) const override
    {
        DAAL_ASSERT(index < size());
        DAAL_ASSERT(probs);
        DAAL_ASSERT(numProbs == _numberOfClasses);
        DAAL_ASSERT(_counters);

        size_t total        = 0;
        size_t * const cnts = _counters + index * _numberOfClasses;
        for (size_t i = 0; i < _numberOfClasses; ++i)
        {
            total += cnts[i];
        }

        if (total != 0)
        {
            for (size_t i = 0; i < _numberOfClasses; ++i)
            {
                probs[i] = static_cast<double>(cnts[i]) / static_cast<double>(total);
            }
        }
        else
        {
            for (size_t i = 0; i < _numberOfClasses; ++i)
            {
                probs[i] = 0.0;
            }
        }
    }

    using BaseType::size;

protected:
    void resetCounters()
    {
        const size_t cnt = size() * _numberOfClasses;
        for (size_t i = 0; i < cnt; ++i)
        {
            _counters[i] = 0;
        }
    }

    size_t majorityClass(size_t index) const
    {
        DAAL_ASSERT(index < size());
        const size_t * const c = &_counters[index * _numberOfClasses];
        return maxElement<cpu>(c, &c[_numberOfClasses]) - c;
    }

private:
    size_t _numberOfClasses;
    size_t * _counters;
};

template <typename algorithmFPType, CpuType cpu, typename LeavesData>
struct CopyNodeContext
{
    const decision_tree::internal::Tree<cpu, algorithmFPType, int> & src;
    decision_tree::classification::DecisionTreeNode * dest;
    double * impVals;
    int * smplCntVals;
    size_t & destNodeCount;
    size_t destNodeCapacity;
    const decision_tree::internal::PruningData<cpu, int> & pruningData;
    const LeavesData & leavesData;
    int & leafIndex;
    int * probIndices;
    double * probs;
    size_t nClasses;
};

template <typename algorithmFPType, CpuType cpu, typename LeavesData>
static void copyNode(const CopyNodeContext<algorithmFPType, cpu, LeavesData> & context, size_t srcNodeIdx, size_t destNodeIdx)
{
    if (context.src[srcNodeIdx].isLeaf())
    {
        context.dest[destNodeIdx].dimension        = static_cast<size_t>(-1);
        context.dest[destNodeIdx].leftIndexOrClass = context.src[srcNodeIdx].dependentVariable();
        context.dest[destNodeIdx].cutPoint         = 0;
        context.impVals[destNodeIdx]               = context.src[srcNodeIdx].impurity();
        context.smplCntVals[destNodeIdx]           = context.src[srcNodeIdx].count();
        context.probIndices[destNodeIdx]           = context.leafIndex;
        context.leavesData.putProbabilities(context.src[srcNodeIdx].leavesDataIndex(), context.probs + context.leafIndex * context.nClasses,
                                            context.nClasses);
        ++context.leafIndex;
    }
    else if (context.pruningData.isPruned(srcNodeIdx))
    {
        context.dest[destNodeIdx].dimension        = static_cast<size_t>(-1);
        context.dest[destNodeIdx].leftIndexOrClass = context.pruningData.dependentVariable(srcNodeIdx);
        context.dest[destNodeIdx].cutPoint         = 0;
        context.impVals[destNodeIdx]               = context.src[srcNodeIdx].impurity();
        context.smplCntVals[destNodeIdx]           = context.src[srcNodeIdx].count();
        context.probIndices[destNodeIdx]           = context.leafIndex;
        context.pruningData.putProbabilities(srcNodeIdx, context.probs + context.leafIndex * context.nClasses, context.nClasses);
        ++context.leafIndex;
    }
    else
    {
        DAAL_ASSERT(context.destNodeCount + 2 <= context.destNodeCapacity);
        context.dest[destNodeIdx].dimension        = context.src[srcNodeIdx].featureIndex();
        const size_t childIndex                    = context.destNodeCount;
        context.dest[destNodeIdx].leftIndexOrClass = childIndex;
        context.dest[destNodeIdx].cutPoint         = context.src[srcNodeIdx].cutPoint();
        context.impVals[destNodeIdx]               = context.src[srcNodeIdx].impurity();
        context.smplCntVals[destNodeIdx]           = context.src[srcNodeIdx].count();
        context.probIndices[destNodeIdx]           = -1;
        context.destNodeCount += 2;
        copyNode(context, context.src[srcNodeIdx].leftChildIndex(), childIndex);
        copyNode(context, context.src[srcNodeIdx].rightChildIndex(), childIndex + 1);
    }
}

template <typename algorithmFPType, CpuType cpu, typename ParameterType, typename LeavesData>
static services::Status pruneAndConvertTree(const NumericTable * px, const NumericTable * py, decision_tree::classification::Model & r,
                                            const ParameterType & parameter, decision_tree::internal::Tree<cpu, algorithmFPType, int> & tree,
                                            const LeavesData & leavesData)
{
    using services::SharedPtr;
    using data_management::HomogenNumericTable;

    services::Status status;
    if (parameter.pruning == reducedErrorPruning)
    {
        DAAL_ASSERT(px);
        DAAL_ASSERT(py);
        REPPruningData<cpu> repData(tree.nodeCount(), parameter.nClasses);
        tree.reducedErrorPruning(*px, *py, repData);

        const size_t nodeCapacity      = countNodes(0, tree, repData);
        const size_t decisionNodeCount = (nodeCapacity > 0) ? (nodeCapacity - 1) / 2 : 0;
        const size_t leafNodeCount     = nodeCapacity - decisionNodeCount;
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status);
        SharedPtr<HomogenNumericTableCPU<double, cpu> > impTbl(new HomogenNumericTableCPU<double, cpu>(1, nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<int, cpu> > smplCntTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<int, cpu> > probIndicesTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<double, cpu> > probTbl(new HomogenNumericTableCPU<double, cpu>(parameter.nClasses, leafNodeCount, status));
        DAAL_CHECK_STATUS_VAR(status);
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double * const impVals         = impTbl->getArray();
        int * const smplCntVals        = smplCntTbl->getArray();
        int * const probIndices        = probIndicesTbl->getArray();
        double * const probs           = probTbl->getArray();
        size_t nodeCount               = 1;
        int leafIndex                  = 0;
        const CopyNodeContext<algorithmFPType, cpu, LeavesData> context {
            tree, nodes, impVals, smplCntVals, nodeCount, nodeCapacity, repData, leavesData, leafIndex, probIndices, probs, parameter.nClasses
        };
        copyNode<>(context, 0, 0);
        DAAL_ASSERT(leafIndex == leafNodeCount);
        r.impl()->setTreeTable(treeTable);
        r.impl()->setImpTable(impTbl);
        r.impl()->setNodeSmplCntTable(smplCntTbl);
        r.impl()->setProbIndicesTable(probIndicesTbl);
        r.impl()->setProbTable(probTbl);
    }
    else
    {
        const size_t nodeCount         = tree.nodeCount();
        const size_t decisionNodeCount = (nodeCount > 0) ? (nodeCount - 1) / 2 : 0;
        const size_t leafNodeCount     = nodeCount - decisionNodeCount;
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status);
        SharedPtr<HomogenNumericTableCPU<double, cpu> > impTbl(new HomogenNumericTableCPU<double, cpu>(1, nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<int, cpu> > smplCntTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<int, cpu> > probIndicesTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status)
        SharedPtr<HomogenNumericTableCPU<double, cpu> > probTbl(new HomogenNumericTableCPU<double, cpu>(parameter.nClasses, leafNodeCount, status));
        DAAL_CHECK_STATUS_VAR(status);

        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double * const impVals         = impTbl->getArray();
        int * const smplCntVals        = smplCntTbl->getArray();
        int * const probIndices        = probIndicesTbl->getArray();
        double * const probs           = probTbl->getArray();
        int leafIndex                  = 0;

        for (size_t i = 0; i < nodeCount; ++i)
        {
            if (tree[i].isLeaf())
            {
                nodes[i].dimension        = static_cast<size_t>(-1);
                nodes[i].leftIndexOrClass = tree[i].dependentVariable();
                nodes[i].cutPoint         = 0;
                probIndices[i]            = leafIndex;
                DAAL_ASSERT(leafIndex >= 0 && leafIndex < leafNodeCount);
                leavesData.putProbabilities(tree[i].leavesDataIndex(), probs + leafIndex * parameter.nClasses, parameter.nClasses);
                ++leafIndex;
            }
            else
            {
                nodes[i].dimension        = tree[i].featureIndex();
                nodes[i].leftIndexOrClass = tree[i].leftChildIndex();
                nodes[i].cutPoint         = tree[i].cutPoint();
                probIndices[i]            = -1;
            }
            impVals[i]     = tree[i].impurity();
            smplCntVals[i] = tree[i].count();
        }
        r.impl()->setTreeTable(treeTable);
        r.impl()->setImpTable(impTbl);
        r.impl()->setNodeSmplCntTable(smplCntTbl);
        r.impl()->setProbIndicesTable(probIndicesTbl);
        r.impl()->setProbTable(probTbl);
    }
    return status;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status DecisionTreeTrainBatchKernel<algorithmFPType, ParameterType, training::defaultDense, cpu>::compute(
    const NumericTable * x, const NumericTable * y, const NumericTable * w, const NumericTable * px, const NumericTable * py,
    decision_tree::classification::Model * r, const ParameterType * parameter)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT(r);
    DAAL_ASSERT(parameter);

    r->setNFeatures(x->getNumberOfColumns());

    services::Status status;
    Tree<cpu, algorithmFPType, int> tree;
    if (w == nullptr)
    {
        LeavesData<cpu, ClassCounters<cpu> > leavesData;
        if (parameter->splitCriterion == gini)
        {
            Gini<algorithmFPType, cpu> splitCriterion;
            status = tree.train(splitCriterion, leavesData, *x, *y, w, parameter->nClasses, parameter->maxTreeDepth,
                                parameter->minObservationsInLeafNodes);
            DAAL_CHECK_STATUS_VAR(status)
            status = pruneAndConvertTree<>(px, py, *r, *parameter, tree, leavesData);
            DAAL_CHECK_STATUS_VAR(status)
        }
        else
        {
            InfoGain<algorithmFPType, cpu> splitCriterion;
            status = tree.train(splitCriterion, leavesData, *x, *y, w, parameter->nClasses, parameter->maxTreeDepth,
                                parameter->minObservationsInLeafNodes);
            DAAL_CHECK_STATUS_VAR(status)
            status = pruneAndConvertTree<>(px, py, *r, *parameter, tree, leavesData);
            DAAL_CHECK_STATUS_VAR(status)
        }
    }
    else
    {
        LeavesData<cpu, ClassWeightsCounters<algorithmFPType, cpu> > leavesData;
        if (parameter->splitCriterion == gini)
        {
            GiniWeighted<algorithmFPType, cpu> splitCriterion;
            status = tree.train(splitCriterion, leavesData, *x, *y, w, parameter->nClasses, parameter->maxTreeDepth,
                                parameter->minObservationsInLeafNodes);
            DAAL_CHECK_STATUS_VAR(status)
            status = pruneAndConvertTree<>(px, py, *r, *parameter, tree, leavesData);
            DAAL_CHECK_STATUS_VAR(status)
        }
        else
        {
            InfoGainWeighted<algorithmFPType, cpu> splitCriterion;
            status = tree.train(splitCriterion, leavesData, *x, *y, w, parameter->nClasses, parameter->maxTreeDepth,
                                parameter->minObservationsInLeafNodes);
            DAAL_CHECK_STATUS_VAR(status)
            status = pruneAndConvertTree<>(px, py, *r, *parameter, tree, leavesData);
            DAAL_CHECK_STATUS_VAR(status)
        }
    }
    return status;
}

} // namespace internal
} // namespace training
} // namespace classification

namespace internal
{
template <CpuType cpu, typename algorithmFPType>
struct CutPointFinder<cpu, algorithmFPType, GiniWeighted<algorithmFPType, cpu> > : private WeightedBaseCutPointFinder<cpu>
{
    using WeightedBaseCutPointFinder<cpu>::find;
};

template <CpuType cpu, typename algorithmFPType>
struct CutPointFinder<cpu, algorithmFPType, InfoGainWeighted<algorithmFPType, cpu> > : private WeightedBaseCutPointFinder<cpu>
{
    using WeightedBaseCutPointFinder<cpu>::find;
};
} // namespace internal

} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
