/* file: gbt_train_partition.i */
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
//  Implementation of partitions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_PARTITION_I__
#define __GBT_TRAIN_PARTITION_I__

#include "src/algorithms/dtrees/gbt/gbt_train_aux.i"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"

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
class PartitionTaskBase : public GbtTask
{
public:
    using SharedDataType  = SharedDataForTree<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using NodeInfoType    = SplitJob<algorithmFPType, cpu>;
    using ImpurityType    = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType   = SplitData<algorithmFPType, ImpurityType>;
    using IndexTypeVector = TVector<RowIndexType, cpu>;

    PartitionTaskBase(size_t iFeature, DAAL_INT idxFeatureValueBestSplit, SharedDataType & sharedData, const NodeInfoType & nodeInfo,
                      SplitDataType & bestSplit)
        : _iFeature(iFeature),
          _idxFeatureValueBestSplit(idxFeatureValueBestSplit),
          _sharedData(sharedData),
          _nodeInfo(nodeInfo),
          _bestSplit(bestSplit)
    {}

    virtual GbtTask * execute()
    {
        int * bestSplitIdx = _sharedData.bestSplitIdxBuf + _nodeInfo.iStart;
        int * aIdx         = _sharedData.aIdx + _nodeInfo.iStart;

        bool bCopy = true;
        if (_idxFeatureValueBestSplit >= 0)
        {
            finalizeBestSplit(_nodeInfo.n, _nodeInfo.iStart);
        }
        else if (_bestSplit.featureUnordered)
        {
            if (_bestSplit.iStart)
            {
                DAAL_ASSERT(_bestSplit.iStart + _bestSplit.nLeft <= _nodeInfo.n);
                services::internal::tmemcpy<int, cpu>(aIdx, bestSplitIdx + _bestSplit.iStart, _bestSplit.nLeft);
                aIdx += _bestSplit.nLeft;
                services::internal::tmemcpy<int, cpu>(aIdx, bestSplitIdx, _bestSplit.iStart);
                aIdx += _bestSplit.iStart;
                bestSplitIdx += _bestSplit.iStart + _bestSplit.nLeft;
                if (_nodeInfo.n > (_bestSplit.iStart + _bestSplit.nLeft))
                    services::internal::tmemcpy<int, cpu>(aIdx, bestSplitIdx, _nodeInfo.n - _bestSplit.iStart - _bestSplit.nLeft);
                bCopy = false;
            }
        }
        if (bCopy && _sharedData.ctx.par().memorySavingMode) services::internal::tmemcpy<int, cpu>(aIdx, bestSplitIdx, _nodeInfo.n);

        return nullptr;
    }

protected:
    virtual void finalizeBestSplit(size_t n, size_t iStart) = 0;

    size_t _iFeature;
    DAAL_INT _idxFeatureValueBestSplit;
    SharedDataType & _sharedData;
    const NodeInfoType & _nodeInfo;
    SplitDataType & _bestSplit;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class PartitionMemSafetyTask : public PartitionTaskBase<algorithmFPType, RowIndexType, BinIndexType, cpu>
{
public:
    using super          = PartitionTaskBase<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using SharedDataType = typename super::SharedDataType;
    using NodeInfoType   = typename super::NodeInfoType;
    using SplitDataType  = typename super::SplitDataType;

    PartitionMemSafetyTask(size_t iFeature, DAAL_INT idxFeatureValueBestSplit, SharedDataType & sharedData, const NodeInfoType & nodeInfo,
                           SplitDataType & bestSplit)
        : super(iFeature, idxFeatureValueBestSplit, sharedData, nodeInfo, bestSplit)
    {}

    virtual void finalizeBestSplit(size_t n, size_t iStart)
    {
        // nothing
    }
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class DefaultPartitionTask : public PartitionTaskBase<algorithmFPType, RowIndexType, BinIndexType, cpu>
{
public:
    using super           = PartitionTaskBase<algorithmFPType, RowIndexType, BinIndexType, cpu>;
    using SharedDataType  = typename super::SharedDataType;
    using NodeInfoType    = typename super::NodeInfoType;
    using ImpurityType    = typename super::ImpurityType;
    using SplitDataType   = typename super::SplitDataType;
    using IndexTypeVector = typename super::IndexTypeVector;

    DefaultPartitionTask(size_t iFeature, DAAL_INT idxFeatureValueBestSplit, SharedDataType & sharedData, const NodeInfoType & nodeInfo,
                         SplitDataType & bestSplit)
        : super(iFeature, idxFeatureValueBestSplit, sharedData, nodeInfo, bestSplit)
    {}

protected:
    virtual void finalizeBestSplit(size_t n, size_t iStart)
    {
        DAAL_ASSERT(_bestSplit.nLeft > 0);
        const DAAL_INT iRowSplitVal = doPartition(n, iStart, _bestSplit, _iFeature, _idxFeatureValueBestSplit);
        DAAL_ASSERT(iRowSplitVal >= 0);
        _bestSplit.iStart = 0;
        if (_sharedData.ctx.dataHelper().indexedFeatures().isBinned(_iFeature))
            _bestSplit.featureValue =
                (algorithmFPType)_sharedData.ctx.dataHelper().indexedFeatures().binRightBorder(_iFeature, _idxFeatureValueBestSplit);
        else
            _bestSplit.featureValue = _sharedData.ctx.dataHelper().getValue(_iFeature, iRowSplitVal);
    }

    DAAL_INT doPartition(size_t n, size_t iStart, SplitDataType & split, DAAL_INT iFeature, size_t idxFeatureValueBestSplit)
    {
        return doPartitionIdx(n, _sharedData.aIdx + iStart, _sharedData.ctx.dataHelper().indexedFeatures().data(iFeature), split.featureUnordered,
                              idxFeatureValueBestSplit, _sharedData.bestSplitIdxBuf + (2 * iStart), split.nLeft);
    }

    DAAL_INT doPartitionIdx(IndexType n, RowIndexType * aIdx, const RowIndexType * indexedFeature, bool featureUnordered,
                            RowIndexType idxFeatureValueBestSplit, RowIndexType * buffer, RowIndexType nLeft)
    {
        DAAL_INT iRowSplitVal = -1;

        const size_t maxNBlocks = 56;
        size_t sizeOfBlock      = 2048;
        size_t nBlocks          = n / sizeOfBlock;
        nBlocks += !!(n - nBlocks * sizeOfBlock);

        if (nBlocks > maxNBlocks)
        {
            nBlocks     = maxNBlocks;
            sizeOfBlock = n / nBlocks + !!(n % nBlocks);
        }

        RowIndexType part_high_left[maxNBlocks];
        RowIndexType part_high_right[maxNBlocks];

        LoopHelper<cpu>::run(true, nBlocks, [&](size_t iBlock) {
            RowIndexType iLeft  = 0;
            RowIndexType iRight = 0;
            const size_t iStart = iBlock * sizeOfBlock;
            const size_t iEnd   = (((iBlock + 1) * sizeOfBlock > n) ? n : iStart + sizeOfBlock);

            RowIndexType * bestSplitIdx      = buffer + 2 * iStart;
            RowIndexType * bestSplitIdxRight = bestSplitIdx + iEnd - iStart;

            if (featureUnordered)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (IndexType i = iStart; i < iEnd; ++i)
                {
                    if (indexedFeature[aIdx[i]] != idxFeatureValueBestSplit)
                        bestSplitIdxRight[iRight++] = aIdx[i];
                    else
                        bestSplitIdx[iLeft++] = aIdx[i];
                }
            }
            else
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (IndexType i = iStart; i < iEnd; ++i)
                {
                    if (indexedFeature[aIdx[i]] > idxFeatureValueBestSplit)
                        bestSplitIdxRight[iRight++] = aIdx[i];
                    else
                        bestSplitIdx[iLeft++] = aIdx[i];
                }
            }

            part_high_left[iBlock]  = iLeft;
            part_high_right[iBlock] = iRight;
        });

        LoopHelper<cpu>::run(true, nBlocks, [&](size_t iBlock) {
            const size_t iStart = iBlock * sizeOfBlock;
            const size_t iEnd   = (((iBlock + 1) * sizeOfBlock > n) ? n : iStart + sizeOfBlock);

            const RowIndexType sizeL = part_high_left[iBlock];
            const RowIndexType sizeR = part_high_right[iBlock];

            RowIndexType offsetLeft  = 0;
            RowIndexType offsetRight = 0;

            for (IndexType i = 0; i < iBlock; ++i)
            {
                offsetLeft += part_high_left[i];
                offsetRight += part_high_right[i];
            }

            RowIndexType * bestSplitIdx      = buffer + 2 * iStart;
            RowIndexType * bestSplitIdxRight = bestSplitIdx + iEnd - iStart;

            services::internal::tmemcpy<IndexType, cpu>(aIdx + nLeft + offsetRight, bestSplitIdxRight, sizeR);
            services::internal::tmemcpy<IndexType, cpu>(aIdx + offsetLeft, bestSplitIdx, sizeL);
        });

        RowIndexType i = 0;
        while (indexedFeature[aIdx[i]] != idxFeatureValueBestSplit) i++;
        iRowSplitVal = aIdx[i];

        return iRowSplitVal;
    }

    using super::_iFeature;
    using super::_idxFeatureValueBestSplit;
    using super::_sharedData;
    using super::_nodeInfo;
    using super::_bestSplit;
};

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
