/* file: gbt_train_split_hist.i */
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
//  Implementation of histogram method for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_SPLIT_HIST_I__
#define __GBT_TRAIN_SPLIT_HIST_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_train_aux.i"
#include "src/services/service_defines.h"
#include "src/algorithms/dtrees/gbt/gbt_train_hist_kernel.i"

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
template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class TreeBuilder;

namespace hist
{
template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename GHSumType, CpuType cpu>
class GHSumsHelper;
template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename ImpurityType, typename GHSumType, typename SplitType,
          typename ResultType, CpuType cpu>
class MaxImpurityDecreaseHelper;
template <typename RowIndexType, typename BinIndexType, typename algorithmFPType, CpuType cpu>
struct ComputeGHSumByRows;
template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
struct MergeGHSums;

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class SplitTaskByColumns : public GbtTask
{
public:
    using SharedDataType = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType   = SplitJob<algorithmFPType, cpu>;
    using ResultType     = Result<algorithmFPType, cpu>;
    using ImpurityType   = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType  = SplitData<algorithmFPType, ImpurityType>;
    using BestSplitType  = typename TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::BestSplit;
    using GHSumType      = ghSum<algorithmFPType, cpu>;
    using MaxImpurityDecrease =
        MaxImpurityDecreaseHelper<algorithmFPType, RowIndexType, BinIndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums = GHSumsHelper<algorithmFPType, RowIndexType, BinIndexType, GHSumType, cpu>;

    SplitTaskByColumns(size_t iFeature, SharedDataType & data, const NodeInfoType & nodeInfo, BestSplitType & bestSplit, ResultType & res)
        : _iFeature(iFeature), _data(data), _node(nodeInfo), _bestSplit(bestSplit), _res(res)
    {}

    virtual GbtTask * execute()
    {
        _res.ghSums   = nullptr;
        _res.isFailed = true;
        computeGHSums();

        if (!_data.ctx.dataHelper().hasDiffFeatureValues(_iFeature, _data.aIdx + _node.iStart, _node.n))
            return nullptr; //all values of the feature are the same

        const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
        algorithmFPType bestImpDec;
        int iBestFeat;
        _bestSplit.safeGetData(bestImpDec, iBestFeat); // TODO: use them
        SplitDataType split(bestImpDec, featureUnordered);

        DAAL_INT idxFeatureBestSplit = -1;
        auto & par                   = _data.ctx.par();
        MaxImpurityDecrease::find(_node.n, par.minObservationsInLeafNode, par.lambda, split, _res, idxFeatureBestSplit, featureUnordered, _data,
                                  _iFeature);

        if (idxFeatureBestSplit >= 0)
        {
            _bestSplit.update(split, idxFeatureBestSplit, _iFeature);
            _res.isFailed = false;
        }

        return nullptr;
    }

    virtual void computeGHSums()
    {
        const size_t nUnique                = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);
        const RowIndexType * indexedFeature = (RowIndexType *)_data.ctx.dataHelper().indexedFeatures().data(_iFeature);

        auto * aGHSum = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        DAAL_ASSERT(aGHSum); //TODO: return status

        GHSums::fillByZero(nUnique, aGHSum);
        algorithmFPType gTotal = 0, hTotal = 0;

        GHSums::compute(_node.iStart, _node.n, indexedFeature, _data.aIdx, _data.ctx.aSampleToF(), (algorithmFPType *)_data.ctx.grad(_data.iTree),
                        aGHSum, gTotal, hTotal, _node.level);

        _res.ghSums   = aGHSum;
        _res.iFeature = _iFeature;
        _res.nUnique  = nUnique;
        _res.gTotal   = gTotal;
        _res.hTotal   = hTotal;
    }

protected:
    const size_t _iFeature;
    SharedDataType & _data;
    const NodeInfoType & _node;
    ResultType & _res;
    BestSplitType & _bestSplit;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename MergedRType, CpuType cpu>
class FindMaxImpurityDecreaseWithGHSumsReduceTaskMerged : public GbtTask
{
public:
    using SharedDataType = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType   = SplitJob<algorithmFPType, cpu>;
    using ResultType     = Result<algorithmFPType, cpu>;
    using GHSumType      = ghSum<algorithmFPType, cpu>;
    using ImpurityType   = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType  = SplitData<algorithmFPType, ImpurityType>;
    using MaxImpurityDecrease =
        MaxImpurityDecreaseHelper<algorithmFPType, RowIndexType, BinIndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums        = GHSumsHelper<algorithmFPType, RowIndexType, BinIndexType, GHSumType, cpu>;
    using BestSplitType = typename TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::BestSplit;
    using TlsType       = TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>;

    FindMaxImpurityDecreaseWithGHSumsReduceTaskMerged(size_t iFeature, size_t nBlocks, SharedDataType & sharedData, const NodeInfoType & node1,
                                                      const NodeInfoType & node2, BestSplitType & bestSplit1, BestSplitType & bestSplit2,
                                                      const ResultType & prevRes, ResultType & res1, ResultType & res2, algorithmFPType ** results,
                                                      size_t size)
        : _iFeature(iFeature),
          _nBlocks(nBlocks),
          _data(sharedData),
          _node1(node1),
          _node2(node2),
          _bestSplit1(bestSplit1),
          _bestSplit2(bestSplit2),
          _res1(res1),
          _res2(res2),
          _prevRes(prevRes),
          _results(results),
          _size(size)
    {}

    virtual GbtTask * execute()
    {
        const size_t nUnique = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);

        _res1.isFailed = true;
        _res1.ghSums   = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        _res1.gTotal   = 0;
        _res1.hTotal   = 0;
        _res1.iFeature = _iFeature;
        _res1.nUnique  = nUnique;

        const size_t iStart = _data.GH_SUMS_BUF->nUniquesArr[_iFeature];
        const size_t iEnd   = iStart + nUnique;

        MergeGHSums<algorithmFPType, RowIndexType, BinIndexType, cpu>::run(nUnique, iStart, iEnd, _results, _size, _res1);

        daal::threader_for(2, 1, [&](size_t iBlock) {
            if (iBlock == 0)
            {
                // TODO: check for hasDiffFeatureValues()

                const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
                algorithmFPType bestImpDec;
                int iBestFeat;
                _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
                SplitDataType split(bestImpDec, featureUnordered);

                DAAL_INT idxFeatureBestSplit = -1;
                auto & par                   = _data.ctx.par();
                MaxImpurityDecrease::find(_node1.n, par.minObservationsInLeafNode, par.lambda, split, _res1, idxFeatureBestSplit, featureUnordered,
                                          _data, _iFeature);

                if (idxFeatureBestSplit >= 0)
                {
                    _bestSplit1.update(split, idxFeatureBestSplit, _iFeature);
                    _res1.isFailed = false;
                }
            }
            else
            {
                auto * aGHSum = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();

                DAAL_ASSERT(aGHSum); //TODO: return status
                const algorithmFPType gTotal = _prevRes.gTotal - _res1.gTotal;
                const algorithmFPType hTotal = _prevRes.hTotal - _res1.hTotal;

                GHSums::computeDiff(nUnique, _prevRes.ghSums, _res1.ghSums, aGHSum);

                _res2.ghSums   = aGHSum;
                _res2.iFeature = _iFeature;
                _res2.nUnique  = nUnique;
                _res2.gTotal   = gTotal;
                _res2.hTotal   = hTotal;

                // TODO: check for hasDiffFeatureValues()

                const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
                algorithmFPType bestImpDec;
                int iBestFeat;
                _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
                SplitDataType split(bestImpDec, featureUnordered);

                DAAL_INT idxFeatureBestSplit = -1;
                auto & par                   = _data.ctx.par();

                MaxImpurityDecrease::find(_node2.n, par.minObservationsInLeafNode, par.lambda, split, _res2, idxFeatureBestSplit, featureUnordered,
                                          _data, _iFeature);

                if (idxFeatureBestSplit >= 0)
                {
                    _bestSplit2.update(split, idxFeatureBestSplit, _iFeature);
                    _res2.isFailed = false;
                }
            }
        });
        return nullptr;
    }

protected:
    const size_t _iFeature;
    const size_t _nBlocks;
    SharedDataType & _data;
    const NodeInfoType & _node1;
    const NodeInfoType & _node2;
    BestSplitType & _bestSplit1;
    BestSplitType & _bestSplit2;
    ResultType & _res1;
    ResultType & _res2;
    const ResultType & _prevRes;
    algorithmFPType ** _results;
    const size_t _size;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, typename MergedRType, CpuType cpu>
class FindMaxImpurityDecreaseWithGHSumsReduceTask : public GbtTask
{
public:
    using SharedDataType = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType   = SplitJob<algorithmFPType, cpu>;
    using ResultType     = Result<algorithmFPType, cpu>;
    using GHSumType      = ghSum<algorithmFPType, cpu>;
    using ImpurityType   = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType  = SplitData<algorithmFPType, ImpurityType>;
    using MaxImpurityDecrease =
        MaxImpurityDecreaseHelper<algorithmFPType, RowIndexType, BinIndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums        = GHSumsHelper<algorithmFPType, RowIndexType, BinIndexType, GHSumType, cpu>;
    using BestSplitType = typename TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::BestSplit;

    FindMaxImpurityDecreaseWithGHSumsReduceTask(size_t iFeature, size_t nBlocks, SharedDataType & sharedData, const NodeInfoType & node1,
                                                BestSplitType & bestSplit1, ResultType & res1, algorithmFPType ** results, size_t size)
        : _iFeature(iFeature),
          _nBlocks(nBlocks),
          _data(sharedData),
          _node1(node1),
          _bestSplit1(bestSplit1),
          _res1(res1),
          _results(results),
          _size(size)
    {}

    virtual GbtTask * execute()
    {
        const size_t nUnique = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);

        _res1.isFailed = true;
        _res1.ghSums   = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        _res1.gTotal   = 0;
        _res1.hTotal   = 0;
        _res1.iFeature = _iFeature;
        _res1.nUnique  = nUnique;

        const size_t iStart = _data.GH_SUMS_BUF->nUniquesArr[_iFeature];
        const size_t iEnd   = iStart + nUnique;

        MergeGHSums<algorithmFPType, RowIndexType, BinIndexType, cpu>::run(nUnique, iStart, iEnd, _results, _size, _res1);

        // TODO: check for hasDiffFeatureValues()

        const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);

        algorithmFPType bestImpDec;
        int iBestFeat;
        _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
        SplitDataType split(bestImpDec, featureUnordered);

        DAAL_INT idxFeatureBestSplit = -1;
        auto & par                   = _data.ctx.par();
        MaxImpurityDecrease::find(_node1.n, par.minObservationsInLeafNode, par.lambda, split, _res1, idxFeatureBestSplit, featureUnordered, _data,
                                  _iFeature);

        if (idxFeatureBestSplit >= 0)
        {
            _bestSplit1.update(split, idxFeatureBestSplit, _iFeature);
            _res1.isFailed = false;
        }
        return nullptr;
    }

protected:
    const size_t _iFeature;
    const size_t _nBlocks;
    SharedDataType & _data;
    const NodeInfoType & _node1;
    BestSplitType & _bestSplit1;
    ResultType & _res1;
    algorithmFPType ** _results;
    size_t _size;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class ComputeGHSumsByRowsTask : public GbtTask
{
public:
    using GHSumType      = ghSum<algorithmFPType, cpu>;
    using SharedDataType = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType   = SplitJob<algorithmFPType, cpu>;
    using ResultType     = TVector<Result<algorithmFPType, cpu>, cpu, ScalableAllocator<cpu> >;
    using GHSums         = GHSumsHelper<algorithmFPType, RowIndexType, BinIndexType, GHSumType, cpu>;
    using TlsType        = TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>;

    ComputeGHSumsByRowsTask(size_t iBlock, size_t blockSize, SharedDataType & data, const NodeInfoType & nodeInfo, TlsType * res)
        : _iBlock(iBlock), _blockSize(blockSize), _data(data), _node(nodeInfo), _res(res)
    {}

    virtual GbtTask * execute()
    {
        const BinIndexType * indexedFeature = _data.GH_SUMS_BUF->newFI;
        int * aIdx                          = _data.aIdx;
        const RowIndexType nFeatures        = _data.ctx.nFeatures();

        const size_t iStart = _iBlock * _blockSize + _node.iStart;
        const size_t iEnd   = (((_iBlock + 1) * _blockSize > _node.n) ? _node.iStart + _node.n : iStart + _blockSize);

        auto * local               = _res->local();
        GHSumType * aGHSum         = local->ghSum;
        algorithmFPType * aGHSumFP = (algorithmFPType *)local->ghSum;

        if (!local->isInitilized)
        {
            GHSums::fillByZero(_data.GH_SUMS_BUF->nDiffFeatMax, aGHSum);
            local->isInitilized = true;
        }

        algorithmFPType * pgh = (algorithmFPType *)_data.ctx.grad(_data.iTree);
        ComputeGHSumByRows<RowIndexType, BinIndexType, algorithmFPType, cpu>::run(aGHSumFP, indexedFeature, aIdx, pgh, nFeatures, iStart, iEnd,
                                                                                  _node.iStart + _node.n, _data.GH_SUMS_BUF->nUniquesArr.get());
        return nullptr;
    }

protected:
    const size_t _iBlock;
    const size_t _blockSize;
    SharedDataType & _data;
    const NodeInfoType & _node;
    TlsType * _res;
};

} /* namespace hist */
} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
