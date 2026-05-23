/* file: dtrees_regression_predict_dense_default_impl.i */
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
//  Implementation of auxiliary functions for dtrees regression predict algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DTREES_REGRESSION_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __DTREES_REGRESSION_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/services/service_environment.h"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace regression
{
namespace prediction
{
namespace internal
{
using namespace dtrees::internal;
//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTaskBase
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTaskBase
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    PredictRegressionTaskBase(const NumericTable * x, NumericTable * y) : _data(x), _res(y) {}

protected:
    static algorithmFPType predict(const dtrees::internal::DecisionTreeTable & t, const dtrees::internal::FeatureTypes & featTypes,
                                   const algorithmFPType * x)
    {
        const typename dtrees::internal::DecisionTreeNode * pNode =
            dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, featTypes, x);
        DAAL_ASSERT(pNode);

        return pNode ? pNode->featureValueOrResponse : 0.;
    }

    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x)
    {
        algorithmFPType val    = 0;
        const size_t iLastTree = iFirstTree + nTrees;

        for (size_t iTree = iFirstTree; iTree < iLastTree; ++iTree) val += predict(*_aTree[iTree], _featHelper, x);
        return val;
    }
    services::Status run(algorithmFPType factor);

protected:
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const dtrees::internal::DecisionTreeTable *, cpu> _aTree;
    const NumericTable * _data;
    NumericTable * _res;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTaskBase<algorithmFPType, cpu>::run(algorithmFPType factor)
{
    const auto nTreesTotal = _aTree.size();
    const auto treeSize    = _aTree[0]->getNumberOfRows() * sizeof(dtrees::internal::DecisionTreeNode);

    dtrees::prediction::internal::TileDimensions<algorithmFPType> dim(*_data, nTreesTotal, treeSize);
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    services::internal::service_memset<algorithmFPType, cpu>(resBD.get(), 0, dim.nRowsTotal);
    const size_t nThreads = daal::threader_get_threads_number();
    SafeStatus safeStat;
    services::Status s;
    for (size_t iTree = 0; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if (!s) return s;
        size_t nTreesToUse = ((iTree + dim.nTreesInBlock) < nTreesTotal ? dim.nTreesInBlock : (nTreesTotal - iTree));
        daal::threader_for(dim.nDataBlocks, 1, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iBlock * dim.nRowsInBlock : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() + iStartRow;
            if (nRowsToProcess < 2 * nThreads)
            {
                for (size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                    res[iRow] += factor * predictByTrees(iTree, nTreesToUse, xBD.get() + iRow * dim.nCols);
            }
            else
            {
                daal::threader_for(nRowsToProcess, 1,
                                   [&](size_t iRow) { res[iRow] += factor * predictByTrees(iTree, nTreesToUse, xBD.get() + iRow * dim.nCols); });
            }
        });
        s = safeStat.detach();
    }
    return s;
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
