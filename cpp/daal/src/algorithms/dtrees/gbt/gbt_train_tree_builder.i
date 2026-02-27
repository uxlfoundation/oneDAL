/* file: gbt_train_tree_builder.i */
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
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_TREE_BUILDER_I__
#define __GBT_TRAIN_TREE_BUILDER_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_train_aux.i"
#include "src/algorithms/dtrees/gbt/gbt_train_partition.i"
#include "src/algorithms/dtrees/gbt/gbt_train_split_hist.i"
#include "src/algorithms/dtrees/gbt/gbt_train_split_sorting.i"
#include "src/algorithms/dtrees/gbt/gbt_train_node_creator.i"
#include "src/algorithms/dtrees/gbt/gbt_train_updater.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class TreeBuilder : public TreeBuilderBase
{
public:
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu> CommonCtx;
    using MemHelperType = MemHelperBase<algorithmFPType, cpu>;
    typedef typename CommonCtx::DataHelperType DataHelperType;

    typedef gh<algorithmFPType, cpu> ghType;
    typedef ghSum<algorithmFPType, cpu> ghSumType;
    typedef SplitJob<algorithmFPType, cpu> SplitJobType;
    typedef gbt::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;

    class BestSplit
    {
    public:
        BestSplit(SplitDataType & split, Mutex * mt) : _split(split), _mt(mt), _iIndexedFeatureSplitValue(-1), _iFeature(-1) {}
        void safeGetData(algorithmFPType & impDec, int & iFeature)
        {
            if (_mt)
            {
                _mt->lock();
                impDec   = impurityDecrease();
                iFeature = _iFeature;
                _mt->unlock();
            }
            else
            {
                impDec   = impurityDecrease();
                iFeature = _iFeature;
            }
        }
        void update(const SplitDataType & split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if (_mt)
            {
                _mt->lock();
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
                _mt->unlock();
            }
            else
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
        }

        void update(const SplitDataType & split, int iFeature, int * bestSplitIdx, const int * aIdx, size_t n)
        {
            if (_mt)
            {
                _mt->lock();
                if (updateImpl(split, -1, iFeature)) services::internal::tmemcpy<int, cpu>(bestSplitIdx, aIdx, n);
                _mt->unlock();
            }
            else
            {
                if (updateImpl(split, -1, iFeature)) services::internal::tmemcpy<int, cpu>(bestSplitIdx, aIdx, n);
            }
        }

        void getResult(DAAL_INT & ifeature, DAAL_INT & indexedFeatureSplitValue)
        {
            ifeature                 = iFeature();
            indexedFeatureSplitValue = iIndexedFeatureSplitValue();
        }

        int iIndexedFeatureSplitValue() const { return _iIndexedFeatureSplitValue; }
        int iFeature() const { return _iFeature; }
        bool isThreadedMode() const { return _mt != nullptr; }

    private:
        algorithmFPType impurityDecrease() const { return _split.impurityDecrease; }
        bool updateImpl(const SplitDataType & split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if (split.impurityDecrease < impurityDecrease()) return false;

            if (split.impurityDecrease == impurityDecrease())
            {
                if (_iFeature < (int)iFeature) //deterministic way, let the split be the same as in sequential case
                    return false;
            }
            _iFeature = (int)iFeature;
            split.copyTo(_split);
            _iIndexedFeatureSplitValue = iIndexedFeatureSplitValue;
            return true;
        }

    private:
        SplitDataType & _split;
        Mutex * _mt;
        RowIndexType _iIndexedFeatureSplitValue;
        DAAL_INT _iFeature;
    };

    TreeBuilder(CommonCtx & ctx) : _ctx(ctx) {}
    ~TreeBuilder()
    {
        delete _memHelper;
        delete _taskGroup;
    }

    bool isInitialized() const { return !!_aBestSplitIdxBuf.get(); }
    virtual services::Status run(gbt::internal::GbtDecisionTree *& pRes, services::SharedPtr<HomogenNumericTable<double> > & pTblImp,
                                 services::SharedPtr<HomogenNumericTable<int> > & pTblSmplCnt, size_t iTree,
                                 GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF);

    virtual services::Status run(gbt::internal::GbtDecisionTree *& pRes, services::SharedPtr<HomogenNumericTable<double> > & pTblImp,
                                 services::SharedPtr<HomogenNumericTable<int> > & pTblSmplCnt, size_t iTree) DAAL_C11_OVERRIDE
    {
        return services::Status();
    }
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        _aBestSplitIdxBuf.reset(_ctx.nSamples() * 2);
        _aSample.reset(_ctx.nSamples());
        DAAL_CHECK_MALLOC(_aBestSplitIdxBuf.get() && _aSample.get());
        DAAL_CHECK_MALLOC(initMemHelper());
        if (_ctx.isParallelNodes() && !_taskGroup) DAAL_CHECK_MALLOC((_taskGroup = new daal::task_group()));
        return services::Status();
    }
    daal::task_group * taskGroup() { return _taskGroup; }
    static TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu> * create(CommonCtx & ctx);

protected:
    bool initMemHelper();
    //find features to check in the current split node
    const RowIndexType * chooseFeatures()
    {
        if (_ctx.nFeatures() == _ctx.nFeaturesPerNode()) return nullptr;
        RowIndexType * featureSample = _memHelper->getFeatureSampleBuf();
        _ctx.chooseFeatures(featureSample);
        return featureSample;
    }

    class TaskForker : public GbtTask
    {
    public:
        typedef TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu> BuilderType;
        typedef TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu> CommonCtx;

        TaskForker(GbtTask * o, CommonCtx & ctx, BuilderType * builder) : _task(o), _ctx(ctx), _builder(builder) {}

        virtual void operator()() { _builder->buildSplit(_task); }
        virtual GbtTask * execute() { return nullptr; }

    protected:
        CommonCtx & _ctx;
        GbtTask * _task;
        BuilderType * _builder;
    };

    void buildNode(TaskForker & task);
    RowIndexType * bestSplitIdxBuf() const { return _aBestSplitIdxBuf.get(); }
    NodeType::Base * buildRoot(size_t iTree, GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF)
    {
        _iTree                = iTree;
        const size_t nSamples = _ctx.nSamples();
        auto aSample          = _aSample.get();

        if (_ctx.isBagging()) // make a copy
        {
            const RowIndexType * const aSampleToF = _ctx.aSampleToF();
            for (size_t i = 0; i < nSamples; ++i) aSample[i] = aSampleToF[i];
        }
        else // use of all data
        {
            for (size_t i = 0; i < nSamples; ++i) aSample[i] = i;
        }

        ImpurityType imp;
        getInitialImpurity(imp);
        typename NodeType::Base * res = buildLeaf(0, nSamples, 0, imp); // use node creater
        if (res) return res;

        SplitJobType job(0, nSamples, 0, imp, res);
        SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu> data(_ctx, bestSplitIdxBuf(), const_cast<int *>(_aSample.get()),
                                                                                 _memHelper, this->_iTree, _tree, _mtAlloc);
        data.GH_SUMS_BUF = &GH_SUMS_BUF;

        if (_ctx.par().memorySavingMode)
        {
            using Mode    = MemorySafetySplitMode<algorithmFPType, RowIndexType, BinIndexType, cpu>;
            using Updater = UpdaterByColumns<algorithmFPType, RowIndexType, BinIndexType, Mode, cpu>;
            buildSplit(new (service_scalable_calloc<Updater, cpu>(1)) Updater(data, job));
        }
        else if (_ctx.par().splitMethod == gbt::training::exact || _ctx.nFeatures() != _ctx.nFeaturesPerNode())
        {
            using Mode    = ExactSplitMode<algorithmFPType, RowIndexType, BinIndexType, cpu>;
            using Updater = UpdaterByColumns<algorithmFPType, RowIndexType, BinIndexType, Mode, cpu>;
            buildSplit(new (service_scalable_calloc<Updater, cpu>(1)) Updater(data, job));
        }
        else
        {
            using Mode    = InexactSplitMode<algorithmFPType, RowIndexType, BinIndexType, cpu>;
            using Updater = UpdaterByRows<algorithmFPType, RowIndexType, BinIndexType, Mode, cpu>;
            buildSplit(new (service_scalable_calloc<Updater, cpu>(1)) Updater(data, job));
        }

        if (taskGroup()) taskGroup()->wait();

        return res;
    }

    void getInitialImpurity(ImpurityType & val)
    {
        const ghType * pgh = _ctx.grad(this->_iTree);
        auto & G           = val.g;
        auto & H           = val.h;
        G = H = 0;

        const size_t nSamples  = _ctx.nSamples();
        const int * aSampleToF = _ctx.aSampleToF();

        const size_t nThreads = _ctx.numAvailableThreads();
        const size_t nBlocks  = getNBlocksForOpt<cpu>(nThreads, nSamples);
        const bool inParallel = nBlocks > 1;
        daal::services::internal::TArray<algorithmFPType, cpu> gsArr(nBlocks);
        daal::services::internal::TArray<algorithmFPType, cpu> hsArr(nBlocks);
        algorithmFPType * const gs = gsArr.get();
        algorithmFPType * const hs = hsArr.get();
        const size_t nPerBlock     = nSamples / nBlocks;
        const size_t nSurplus      = nSamples % nBlocks;
        LoopHelper<cpu>::run(inParallel, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock + 1 > nSurplus ? nPerBlock * iBlock + nSurplus : (nPerBlock + 1) * iBlock;
            const size_t end   = iBlock + 1 > nSurplus ? start + nPerBlock : start + (nPerBlock + 1);
            algorithmFPType localG, localH;
            localG = localH = 0;
            if (aSampleToF)
            {
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    localG += pgh[aSampleToF[i]].g;
                    localH += pgh[aSampleToF[i]].h;
                }
            }
            else
            {
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    localG += pgh[i].g;
                    localH += pgh[i].h;
                }
            }
            gs[iBlock] = localG;
            hs[iBlock] = localH;
        });
        for (size_t i = 0; i < nBlocks; i++)
        {
            G += gs[i];
            H += hs[i];
        }
    }

    NodeType::Base * buildLeaf(size_t iStart, size_t n, size_t level, const ImpurityType & imp)
    {
        return _ctx.terminateCriteria(n, level, imp) ? makeLeaf(_aSample.get() + iStart, n, imp) : nullptr;
    }

    typename NodeType::Leaf * makeLeaf(const int * idx, size_t n, const ImpurityType & imp)
    {
        typename NodeType::Leaf * pNode = nullptr;
        if (_ctx.isThreaded())
        {
            _mtAlloc.lock();
            pNode = _tree.allocator().allocLeaf();
            _mtAlloc.unlock();
        }
        else
            pNode = _tree.allocator().allocLeaf();
        pNode->response = _ctx.computeLeafWeightUpdateF(idx, n, imp, _iTree);
        pNode->count    = n;
        pNode->impurity = imp.value(_ctx.par().lambda);
        return pNode;
    }
    void buildSplit(GbtTask * task);

protected:
    CommonCtx & _ctx;
    size_t _iTree = 0;
    TreeType _tree;
    daal::Mutex _mtAlloc;
    typedef dtrees::internal::TVector<RowIndexType, cpu> IndexTypeArray;
    mutable IndexTypeArray _aBestSplitIdxBuf;
    mutable IndexTypeArray _aSample;
    MemHelperType * _memHelper    = nullptr;
    daal::task_group * _taskGroup = nullptr;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
bool TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::initMemHelper()
{
    auto featuresSampleBufSize = 0; //do not allocate if not required
    const auto nFeat           = _ctx.nFeatures();
    if (nFeat != _ctx.nFeaturesPerNode())
    {
        // cast to 64-bits uint to avoid potential overflow
        const auto nFeaturesPerNode = static_cast<uint64_t>(_ctx.nFeaturesPerNode());
        if (nFeaturesPerNode * nFeaturesPerNode < 2 * nFeat)
            featuresSampleBufSize = 2 * _ctx.nFeaturesPerNode();
        else
            featuresSampleBufSize = nFeat;
    }
    if (_ctx.isThreaded())
        _memHelper = new MemHelperThr<algorithmFPType, cpu>(featuresSampleBufSize);
    else
        _memHelper = new MemHelperSeq<algorithmFPType, cpu>(
            featuresSampleBufSize, _ctx.par().memorySavingMode ? 0 : _ctx.dataHelper().indexedFeatures().maxNumIndices(), _ctx.nSamples());
    return _memHelper && _memHelper->init();
}

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
void TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::buildNode(TaskForker & task)
{
    if (taskGroup())
        taskGroup()->run(task);
    else
        buildSplit(&task);
}

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
services::Status TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::run(gbt::internal::GbtDecisionTree *& pRes,
                                                                                    services::SharedPtr<HomogenNumericTable<double> > & pTblImp,
                                                                                    services::SharedPtr<HomogenNumericTable<int> > & pTblSmplCnt,
                                                                                    size_t iTree,
                                                                                    GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF)
{
    _tree.destroy();
    typename NodeType::Base * nd = buildRoot(iTree, GH_SUMS_BUF);
    DAAL_CHECK_MALLOC(nd);

    _tree.reset(nd, false);
    services::Status status = gbt::internal::ModelImpl::treeToTable(_tree, &pRes, pTblImp, pTblSmplCnt, _ctx.nFeatures());
    DAAL_CHECK_STATUS_VAR(status)

    if (_ctx.isBagging() && _tree.top()) _ctx.updateOOB(iTree, _tree);

    return services::Status();
}

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
void TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::buildSplit(GbtTask * task)
{
    task->execute();

    GbtTask * newTasks[2];
    size_t nTasks = 0;

    task->getNextTasks(newTasks, nTasks); // returns 0, 1 or 2 tasks

    task->~GbtTask();
    service_scalable_free<GbtTask, cpu>(task);

    if (nTasks == 1)
    {
        buildSplit(newTasks[0]);
    }
    else if (nTasks == 2)
    {
        if (_ctx.numAvailableThreads())
        {
            TaskForker newTask(newTasks[0], _ctx, this);
            buildNode(newTask);
        }
        else
            buildSplit(newTasks[0]);
        buildSplit(newTasks[1]);
    }
}

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu> * TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::create(CommonCtx & ctx)
{
    return new TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>(ctx);
}

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
