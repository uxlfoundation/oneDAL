/* file: decision_tree_regression_train_dense_default_impl.i */
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

#ifndef __DECISION_TREE_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DECISION_TREE_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "services/daal_defines.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/decision_tree/decision_tree_regression_model_impl.h"
#include "src/algorithms/decision_tree/decision_tree_regression_train_kernel.h"
#include "src/algorithms/decision_tree/decision_tree_train_impl.i"
#include "src/algorithms/decision_tree/decision_tree_regression_split_criterion.i"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace decision_tree::internal;

template <typename algorithmFPType, CpuType cpu>
class REPPruningData : public PruningData<cpu, algorithmFPType>
{
    typedef PruningData<cpu, algorithmFPType> BaseType;

public:
    REPPruningData(size_t size) : BaseType(size), _data(daal_alloc<algorithmFPType>(size ? size * 3 : 1)) { resetData(); }

    ~REPPruningData() override
    {
        daal_free(_data);
        _data = nullptr;
    }

    void update(size_t index, algorithmFPType v)
    {
        DAAL_ASSERT(index < size());
        algorithmFPType & count = _data[index * 3];
        algorithmFPType & mean  = _data[index * 3 + 1];
        algorithmFPType & mse   = _data[index * 3 + 2];

        // Welford running method.

        if (++count == 1)
        {
            mean = v;
            mse  = algorithmFPType(0);
        }
        else
        {
            const algorithmFPType delta = v - mean;
            mean += delta / count;
            mse += delta * (v - mean);
        }
    }

    algorithmFPType count(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3];
    }

    algorithmFPType mean(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3 + 1];
    }

    algorithmFPType mse(size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index * 3 + 2];
    }

    typedef algorithmFPType ErrorType;

    ErrorType error(size_t index, algorithmFPType v) const
    {
        const algorithmFPType delta = mean(index) - v;
        return mse(index) + count(index) * delta * delta;
    }

    ErrorType error(size_t index) const { return mse(index); }

    void prune(size_t index) { BaseType::prune(index, mean(index)); }

    using BaseType::size;

protected:
    void resetData()
    {
        const size_t cnt = size() * 3;
        for (size_t i = 0; i < cnt; ++i)
        {
            _data[i] = algorithmFPType(0);
        }
    }

private:
    algorithmFPType * _data;
};

template <typename algorithmFPType, CpuType cpu>
static void copyNode(size_t srcNodeIdx, size_t destNodeIdx, const decision_tree::internal::Tree<cpu, algorithmFPType, algorithmFPType> & src,
                     decision_tree::regression::DecisionTreeNode * dest, double * impVals, int * smplCntVals, size_t & destNodeCount,
                     size_t destNodeCapacity, const decision_tree::internal::PruningData<cpu, algorithmFPType> & pruningData)
{
    if (src[srcNodeIdx].isLeaf())
    {
        dest[destNodeIdx].dimension                   = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndex                   = 0;
        dest[destNodeIdx].cutPointOrDependantVariable = src[srcNodeIdx].dependentVariable();
        impVals[destNodeIdx]                          = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx]                      = src[srcNodeIdx].count();
    }
    else if (pruningData.isPruned(srcNodeIdx))
    {
        dest[destNodeIdx].dimension                   = static_cast<size_t>(-1);
        dest[destNodeIdx].leftIndex                   = 0;
        dest[destNodeIdx].cutPointOrDependantVariable = pruningData.dependentVariable(srcNodeIdx);
        impVals[destNodeIdx]                          = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx]                      = src[srcNodeIdx].count();
    }
    else
    {
        DAAL_ASSERT(destNodeCount + 2 <= destNodeCapacity);
        dest[destNodeIdx].dimension                   = src[srcNodeIdx].featureIndex();
        const size_t childIndex                       = destNodeCount;
        dest[destNodeIdx].leftIndex                   = childIndex;
        dest[destNodeIdx].cutPointOrDependantVariable = src[srcNodeIdx].cutPoint();
        impVals[destNodeIdx]                          = src[srcNodeIdx].impurity();
        smplCntVals[destNodeIdx]                      = src[srcNodeIdx].count();
        destNodeCount += 2;
        copyNode(src[srcNodeIdx].leftChildIndex(), childIndex, src, dest, impVals, smplCntVals, destNodeCount, destNodeCapacity, pruningData);
        copyNode(src[srcNodeIdx].rightChildIndex(), childIndex + 1, src, dest, impVals, smplCntVals, destNodeCount, destNodeCapacity, pruningData);
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status DecisionTreeTrainBatchKernel<algorithmFPType, training::defaultDense, cpu>::compute(const NumericTable * x, const NumericTable * y,
                                                                                                     const NumericTable * w, const NumericTable * px,
                                                                                                     const NumericTable * py,
                                                                                                     decision_tree::regression::Model * r,
                                                                                                     const daal::algorithms::Parameter * par)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT(r);
    services::Status status;
    const decision_tree::regression::Parameter * const parameter = static_cast<const decision_tree::regression::Parameter *>(par);
    DAAL_ASSERT(parameter);

    DAAL_ASSERT(r->impl());
    r->impl()->setNumberOfFeatures(x->getNumberOfColumns());

    Tree<cpu, algorithmFPType, algorithmFPType> tree;
    if (w == nullptr)
    {
        MSE<algorithmFPType, cpu> splitCriterion;
        status = tree.train(splitCriterion, *x, *y, w, 0, parameter->maxTreeDepth, parameter->minObservationsInLeafNodes);
    }
    else
    {
        MSEWeighted<algorithmFPType, cpu> splitCriterion;
        status = tree.train(splitCriterion, *x, *y, w, 0, parameter->maxTreeDepth, parameter->minObservationsInLeafNodes);
    }
    if (parameter->pruning == reducedErrorPruning)
    {
        DAAL_ASSERT(px);
        DAAL_ASSERT(py);
        REPPruningData<algorithmFPType, cpu> repData(tree.nodeCount());
        tree.reducedErrorPruning(*px, *py, repData);

        const size_t nodeCapacity = countNodes(0, tree, repData);
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status)
        services::SharedPtr<HomogenNumericTableCPU<double, cpu> > impTbl(new HomogenNumericTableCPU<double, cpu>(1, nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status)
        services::SharedPtr<HomogenNumericTableCPU<int, cpu> > smplCntTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCapacity, status));
        DAAL_CHECK_STATUS_VAR(status);
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double * impVals               = impTbl->getArray();
        int * smplCntVals              = smplCntTbl->getArray();
        size_t nodeCount               = 1;
        copyNode(0, 0, tree, nodes, impVals, smplCntVals, nodeCount, nodeCapacity, repData);
        r->impl()->setTreeTable(treeTable);
        r->impl()->setImpTable(impTbl);
        r->impl()->setNodeSmplCntTable(smplCntTbl);
    }
    else
    {
        const size_t nodeCount = tree.nodeCount();
        DecisionTreeTablePtr treeTable(new DecisionTreeTable(nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status)
        services::SharedPtr<HomogenNumericTableCPU<double, cpu> > impTbl(new HomogenNumericTableCPU<double, cpu>(1, nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status)
        services::SharedPtr<HomogenNumericTableCPU<int, cpu> > smplCntTbl(new HomogenNumericTableCPU<int, cpu>(1, nodeCount, status));
        DAAL_CHECK_STATUS_VAR(status);
        DecisionTreeNode * const nodes = static_cast<DecisionTreeNode *>(treeTable->getArray());
        double * impVals               = impTbl->getArray();
        int * smplCntVals              = smplCntTbl->getArray();

        for (size_t i = 0; i < nodeCount; ++i)
        {
            if (tree[i].isLeaf())
            {
                nodes[i].dimension                   = static_cast<size_t>(-1);
                nodes[i].leftIndex                   = 0;
                nodes[i].cutPointOrDependantVariable = tree[i].dependentVariable();
            }
            else
            {
                nodes[i].dimension                   = tree[i].featureIndex();
                nodes[i].leftIndex                   = tree[i].leftChildIndex();
                nodes[i].cutPointOrDependantVariable = tree[i].cutPoint();
            }
            impVals[i]     = tree[i].impurity();
            smplCntVals[i] = tree[i].count();
        }
        r->impl()->setTreeTable(treeTable);
        r->impl()->setImpTable(impTbl);
        r->impl()->setNodeSmplCntTable(smplCntTbl);
    }
    return status;
}

} // namespace internal
} // namespace training
} // namespace regression
namespace internal
{
template <CpuType cpu, typename algorithmFPType>
struct CutPointFinder<cpu, algorithmFPType, decision_tree::regression::training::internal::MSEWeighted<algorithmFPType, cpu> >
    : private WeightedBaseCutPointFinder<cpu>
{
    using WeightedBaseCutPointFinder<cpu>::find;
};
} // namespace internal

} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
