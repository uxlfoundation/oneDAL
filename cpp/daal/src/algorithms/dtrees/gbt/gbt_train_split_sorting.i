/* file: gbt_train_split_sorting.i */
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
//  Implementation of sorting method for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_SPLIT_SORTING_I__
#define __GBT_TRAIN_SPLIT_SORTING_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_train_aux.i"

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

namespace sorting
{
template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class SplitTask : public GbtTask
{
public:
    using SharedDataType  = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType    = SplitJob<algorithmFPType, cpu>;
    using ResultType      = EmptyResult<cpu>;
    using ImpurityType    = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType   = SplitData<algorithmFPType, ImpurityType>;
    using BestSplitType   = typename TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::BestSplit;
    using IndexTypeVector = TVector<RowIndexType, cpu>;

    SplitTask(size_t iFeature, SharedDataType & sharedData, const NodeInfoType & nodeInfo, BestSplitType & bestSplit, ResultType & res)
        : _iFeature(iFeature), _sharedData(sharedData), _nodeInfo(nodeInfo), _bestSplit(bestSplit), _res(res)
    {}

    virtual GbtTask * execute()
    {
        RowIndexType * aIdx = _sharedData.aIdx + _nodeInfo.iStart;

        const bool bThreaded        = _bestSplit.isThreadedMode();
        RowIndexType * bestSplitIdx = _sharedData.bestSplitIdxBuf + _nodeInfo.iStart;
        auto aFeatBuf               = _sharedData.memHelper->getFeatureValueBuf(_nodeInfo.n);

        IndexTypeVector * aFeatIdxBuf = nullptr;
        if (bThreaded)
        {
            //get a local index, since it is used by parallel threads
            aFeatIdxBuf = _sharedData.memHelper->getSortedFeatureIdxBuf(_nodeInfo.n);
            services::internal::tmemcpy<RowIndexType, cpu>(aFeatIdxBuf->get(), aIdx, _nodeInfo.n);
            aIdx = aFeatIdxBuf->get();
        }
        algorithmFPType * featBuf = aFeatBuf->get();
        _sharedData.ctx.featureValuesToBuf(_iFeature, featBuf, aIdx, _nodeInfo.n);
        if (featBuf[_nodeInfo.n - 1] - featBuf[0] <= _sharedData.ctx.accuracy()) //all values of the feature are the same
        {
            _sharedData.memHelper->releaseFeatureValueBuf(aFeatBuf);
            if (aFeatIdxBuf) _sharedData.memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
            return nullptr;
        }
        //use best split estimation when searching on iFeature
        algorithmFPType bestImpDec;
        int iBestFeat;
        _bestSplit.safeGetData(bestImpDec, iBestFeat);
        SplitDataType split(bestImpDec, _sharedData.ctx.featTypes().isUnordered(_iFeature));
        bool bFound = findBestSplitFeatSorted(featBuf, aIdx, split, iBestFeat < 0 || iBestFeat > _iFeature);
        _sharedData.memHelper->releaseFeatureValueBuf(aFeatBuf);
        if (bFound)
        {
            DAAL_ASSERT(split.iStart < _nodeInfo.n);
            DAAL_ASSERT(split.iStart + split.nLeft <= _nodeInfo.n);

            _bestSplit.update(split, _iFeature, bestSplitIdx, aIdx, _nodeInfo.n);
        }
        if (aFeatIdxBuf) _sharedData.memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);

        return nullptr;
    }

protected:
    bool findBestSplitFeatSorted(const algorithmFPType * featureVal, const RowIndexType * aIdx, SplitDataType & split, bool bUpdateWhenTie) const
    {
        return split.featureUnordered ? findBestSplitCategorical(featureVal, aIdx, split, bUpdateWhenTie) :
                                        findBestSplitOrdered(featureVal, aIdx, split, bUpdateWhenTie);
    }

    bool findBestSplitOrdered(const algorithmFPType * featureVal, const RowIndexType * aIdx, SplitDataType & split, bool bUpdateWhenTie) const
    {
        ImpurityType left(_sharedData.ctx.grad(_sharedData.iTree)[*aIdx]);
        algorithmFPType bestImpurityDecrease = split.impurityDecrease;
        DAAL_INT iBest                       = -1;
        const size_t n                       = _nodeInfo.n;
        const auto nMinSplitPart             = _sharedData.ctx.par().minObservationsInLeafNode;
        for (size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
        {
            const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + _sharedData.ctx.accuracy());
            if (!(bSameFeaturePrev || i < nMinSplitPart))
            {
                //can make a split
                //nLeft == i, nRight == n - i
                ImpurityType right(_nodeInfo.imp, left);
                const algorithmFPType v = left.value(_sharedData.ctx.par().lambda) + right.value(_sharedData.ctx.par().lambda);
                if ((v > bestImpurityDecrease) || (bUpdateWhenTie && (v == bestImpurityDecrease)))
                {
                    bestImpurityDecrease = v;
                    split.left           = left;
                    iBest                = i;
                }
            }

            //update impurity and continue
            left.add(_sharedData.ctx.grad(_sharedData.iTree)[aIdx[i]]);
        }
        if (iBest < 0) return false;

        split.impurityDecrease = bestImpurityDecrease;
        split.nLeft            = iBest;
        split.iStart           = 0;
        split.featureValue     = featureVal[iBest - 1];
        return true;
    }

    bool findBestSplitCategorical(const algorithmFPType * featureVal, const RowIndexType * aIdx, SplitDataType & split, bool bUpdateWhenTie) const
    {
        const size_t n           = _nodeInfo.n;
        const auto nMinSplitPart = _sharedData.ctx.par().minObservationsInLeafNode;
        // TODO: check if 2 * iStart is related to indexes
        DAAL_ASSERT(n >= 2 * nMinSplitPart);
        algorithmFPType bestImpurityDecrease = split.impurityDecrease;
        ImpurityType left;
        bool bFound               = false;
        size_t nDiffFeatureValues = 0;
        for (size_t i = 0; i < n - nMinSplitPart;)
        {
            ++nDiffFeatureValues;
            size_t count                = 1;
            const algorithmFPType first = featureVal[i];
            const size_t iStart         = i;
            for (++i; (i < n) && (featureVal[i] == first); ++count, ++i)
                ;
            if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart)) continue;

            if ((i == n) && (nDiffFeatureValues == 2) && bFound) break; //only 2 feature values, one possible split, already found

            calcImpurityIndirect(aIdx + iStart, count, left);
            ImpurityType right(_nodeInfo.imp, left);
            const algorithmFPType v = left.value(_sharedData.ctx.par().lambda) + right.value(_sharedData.ctx.par().lambda);
            if (v > bestImpurityDecrease || (bUpdateWhenTie && (v == bestImpurityDecrease)))
            {
                bestImpurityDecrease = v;
                split.left           = left;
                split.nLeft          = count;
                split.iStart         = iStart;
                split.featureValue   = first;
                bFound               = true;
            }
        }
        if (bFound) split.impurityDecrease = bestImpurityDecrease;
        return bFound;
    }

    void calcImpurityIndirect(const RowIndexType * aIdx, size_t n, ImpurityType & imp) const // move to common code
    {
        DAAL_ASSERT(n);
        const algorithmFPType * pgh = (const algorithmFPType *)_sharedData.ctx.grad(_sharedData.iTree);
        imp                         = _sharedData.ctx.grad(_sharedData.iTree)[aIdx[0]];

        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < n; ++i)
        {
            // TODO: fix
            imp.g += pgh[2 * aIdx[i]];
            imp.h += pgh[2 * aIdx[i] + 1];
        }
    }

    const size_t _iFeature;
    SharedDataType & _sharedData;
    const NodeInfoType & _nodeInfo;
    ResultType & _res;
    BestSplitType & _bestSplit;
};

} // namespace sorting
} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
