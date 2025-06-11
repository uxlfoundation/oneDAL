/* file: gbt_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees regression predictions calculation
//--
*/

#ifndef __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_kernel.h"
#include "src/algorithms/dtrees/gbt/treeshap.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/threading/threading.h"
#include <cfloat> // DBL_EPSILON

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace internal
{
using gbt::internal::FeatureIndexType;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask
{
public:
    typedef gbt::internal::GbtDecisionTree TreeType;
    typedef gbt::prediction::internal::TileDimensions<algorithmFPType> DimType;
    PredictRegressionTask(const NumericTable * x, NumericTable * y) : _data(x), _res(y) {}

    services::Status run(const gbt::regression::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp,
                         bool predShapContributions, bool predShapInteractions);

protected:
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    using dispatcher_t = gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing>;

    services::Status runInternal(services::HostAppIface * pHostApp, NumericTable * result, double predictionBias, bool predShapContributions,
                                 bool predShapInteractions);
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                   const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);
    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    void predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x, algorithmFPType * res,
                              const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);

    inline size_t getNumberOfNodes(size_t nTrees)
    {
        size_t nNodesTotal = 0ul;
        for (size_t iTree = 0ul; iTree < nTrees; ++iTree)
        {
            nNodesTotal += _aTree[iTree]->getNumberOfNodes();
        }
        return nNodesTotal;
    }

    inline bool checkForMissing(const algorithmFPType * x, size_t nTrees, size_t nRows, size_t nColumns) const
    {
        size_t nLvlTotal = 0ul;
        for (size_t iTree = 0ul; iTree < nTrees; ++iTree)
        {
            nLvlTotal += _aTree[iTree]->getMaxLvl();
        }
        if (nLvlTotal <= nColumns)
        {
            // Checking is complicated. Better to do it during inference
            return true;
        }
        else
        {
            for (size_t idx = 0ul; idx < nRows * nColumns; ++idx)
            {
                if (checkFinitenessByComparison(x[idx])) return true;
            }
        }
        return false;
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        size_t iRow;
        dispatcher_t<hasUnorderedFeatures, hasAnyMissing> dispatcher;
        for (iRow = 0; iRow + vectorBlockSize <= nRows; iRow += vectorBlockSize)
        {
            predictByTreesVector<hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(iTree, nTrees, x + iRow * nColumns, res + iRow, dispatcher);
        }
        for (; iRow < nRows; ++iRow)
        {
            res[iRow] += predictByTrees(iTree, nTrees, x + iRow * nColumns, dispatcher);
        }
    }

    template <bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            predict<true, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<false, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
    }

    template <size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRows, nColumns);
        if (hasAnyMissing)
        {
            predict<true, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<false, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    inline services::Status predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x, algorithmFPType * res,
                                                 int condition, FeatureIndexType conditionFeature, const DimType & dim);

    template <bool hasAnyMissing>
    inline services::Status predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x, algorithmFPType * res,
                                                 int condition, FeatureIndexType conditionFeature, const DimType & dim)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            return predictContributions<true, hasAnyMissing>(iTree, nTrees, nRowsData, x, res, condition, conditionFeature, dim);
        }
        else
        {
            return predictContributions<false, hasAnyMissing>(iTree, nTrees, nRowsData, x, res, condition, conditionFeature, dim);
        }
    }

    // TODO: Add vectorBlockSize templates, similar to predict
    // template <size_t vectorBlockSize>
    inline services::Status predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x, algorithmFPType * res,
                                                 int condition, FeatureIndexType conditionFeature, const DimType & dim)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRowsData, dim.nCols);
        if (hasAnyMissing)
        {
            return predictContributions<true>(iTree, nTrees, nRowsData, x, res, condition, conditionFeature, dim);
        }
        else
        {
            return predictContributions<false>(iTree, nTrees, nRowsData, x, res, condition, conditionFeature, dim);
        }
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    inline services::Status predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x,
                                                            const algorithmFPType * nominal, algorithmFPType * res, const DimType & dim);

    template <bool hasAnyMissing>
    inline services::Status predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x,
                                                            const algorithmFPType * nominal, algorithmFPType * res, const DimType & dim)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            return predictContributionInteractions<true, hasAnyMissing>(iTree, nTrees, nRowsData, x, nominal, res, dim);
        }
        else
        {
            return predictContributionInteractions<false, hasAnyMissing>(iTree, nTrees, nRowsData, x, nominal, res, dim);
        }
    }

    // TODO: Add vectorBlockSize templates, similar to predict
    // template <size_t vectorBlockSize>
    inline services::Status predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, const algorithmFPType * x,
                                                            const algorithmFPType * nominal, algorithmFPType * res, const DimType & dim)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRowsData, dim.nCols);
        if (hasAnyMissing)
        {
            return predictContributionInteractions<true>(iTree, nTrees, nRowsData, x, nominal, res, dim);
        }
        else
        {
            return predictContributionInteractions<false>(iTree, nTrees, nRowsData, x, nominal, res, dim);
        }
    }

    template <bool val>
    struct BooleanConstant
    {
        typedef BooleanConstant<val> type;
    };

    // Recursively checking template parameter until it becomes equal to dim.vectorBlockSizeFactor or equal to DimType::minVectorBlockSizeFactor.
    template <size_t vectorBlockSizeFactor>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, const algorithmFPType * x, algorithmFPType * res, const DimType & dim,
                        BooleanConstant<true> keepLooking)
    {
        const size_t nColumns                = dim.nCols;
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        if (dim.vectorBlockSizeFactor == vectorBlockSizeFactor)
        {
            predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<vectorBlockSizeFactor - 1>(iTree, nTrees, nRows, x, res, dim,
                                               BooleanConstant<vectorBlockSizeFactor != DimType::minVectorBlockSizeFactor>());
        }
    }

    template <size_t vectorBlockSizeFactor>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, const algorithmFPType * x, algorithmFPType * res, const DimType & dim,
                        BooleanConstant<false> keepLooking)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, dim.nCols, x, res);
    }

    inline void predict(size_t iTree, size_t nTrees, size_t nRows, const algorithmFPType * x, algorithmFPType * res, const DimType & dim)
    {
        const size_t nColumns                     = dim.nCols;
        constexpr size_t maxVectorBlockSizeFactor = DimType::maxVectorBlockSizeFactor;
        if (maxVectorBlockSizeFactor > 1)
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, x, res, dim, BooleanConstant<true>());
        }
        else
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, x, res, dim, BooleanConstant<false>());
        }
    }

protected:
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const TreeType *, cpu> _aTree;
    const NumericTable * _data;
    NumericTable * _res;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const regression::Model * m, NumericTable * r, size_t nIterations,
                                                                      bool predShapContributions, bool predShapInteractions)
{
    const daal::algorithms::gbt::regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::regression::internal::ModelImpl *>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, nIterations, pHostApp, predShapContributions, predShapInteractions);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(const gbt::regression::internal::ModelImpl * m, size_t nIterations,
                                                                  services::HostAppIface * pHostApp, bool predShapContributions,
                                                                  bool predShapInteractions)
{
    DAAL_ASSERT(nIterations || nIterations <= m->size());
    DAAL_CHECK_MALLOC(_featHelper.init(*_data));
    const auto nTreesTotal = (nIterations ? nIterations : m->size());
    _aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(_aTree.get());

    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0ul; i < nTreesTotal; ++i) _aTree[i] = m->at(i);

    return runInternal(pHostApp, this->_res, m->getPredictionBias(), predShapContributions, predShapInteractions);
}

/**
 * Helper to predict SHAP contribution values
 * \param[in] iTree index of start tree for the calculation
 * \param[in] nTrees number of trees in block included in calculation
 * \param[in] nRowsData number of rows to process
 * \param[in] x pointer to the start of observation data
 * \param[out] res pointer to the start of memory where results are written to
 * \param[in] condition condition the feature to on (1) off (-1) or unconditioned (0)
 * \param[in] conditionFeature index of feature that should be conditioned
 * \param[in] dim DimType helper
*/
template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
services::Status PredictRegressionTask<algorithmFPType, cpu>::predictContributions(size_t iTree, size_t nTrees, size_t nRowsData,
                                                                                   const algorithmFPType * x, algorithmFPType * res, int condition,
                                                                                   FeatureIndexType conditionFeature, const DimType & dim)
{
    // TODO: Make use of vectorBlockSize, similar to predictByTreesVector
    Status st;

    const size_t nColumnsData  = dim.nCols;
    const size_t nColumnsPhi   = nColumnsData + 1;
    const size_t biasTermIndex = nColumnsPhi - 1;

    for (size_t iRow = 0ul; iRow < nRowsData; ++iRow)
    {
        const algorithmFPType * const currentX = x + (iRow * nColumnsData);
        algorithmFPType * const phi            = res + (iRow * nColumnsPhi);
        for (size_t currentTreeIndex = iTree; currentTreeIndex < iTree + nTrees; ++currentTreeIndex)
        {
            const gbt::internal::GbtDecisionTree * currentTree = _aTree[currentTreeIndex];
            st |= gbt::treeshap::treeShap<algorithmFPType, cpu, hasUnorderedFeatures, hasAnyMissing>(currentTree, currentX, phi, 1, &_featHelper,
                                                                                                     condition, conditionFeature);
        }

        if (condition == 0)
        {
            // find bias term by leveraging bias = nominal - sum_i phi_i
            for (int iFeature = 0; iFeature < nColumnsData; ++iFeature)
            {
                phi[biasTermIndex] -= phi[iFeature];
            }
        }
    }

    return st;
}

/**
 * Helper to predict SHAP contribution interactions
 * \param[in] iTree index of start tree for the calculation
 * \param[in] nTrees number of trees in block included in calculation
 * \param[in] nRowsData number of rows to process
 * \param[in] nColumnsData number of columns in data, i.e. features
 *                         note: 1 SHAP value per feature + bias term
 * \param[in] x pointer to the start of observation data
 * \param[out] res pointer to the start of memory where results are written to
 * \param[in] dim DimType helper
*/
template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
services::Status PredictRegressionTask<algorithmFPType, cpu>::predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData,
                                                                                              const algorithmFPType * x,
                                                                                              const algorithmFPType * nominal, algorithmFPType * res,
                                                                                              const DimType & dim)
{
    Status st;
    const size_t nColumnsData  = dim.nCols;
    const size_t nColumnsPhi   = nColumnsData + 1;
    const size_t biasTermIndex = nColumnsPhi - 1;

    const size_t interactionMatrixSize = nColumnsPhi * nColumnsPhi;

    // Allocate buffer for 3 matrices for algorithmFPType of size (nRowsData, nColumnsData)
    const size_t elementsInMatrix = nRowsData * nColumnsPhi;
    const size_t bufferSize       = 3 * sizeof(algorithmFPType) * elementsInMatrix;
    algorithmFPType * buffer      = static_cast<algorithmFPType *>(daal_calloc(bufferSize));
    if (!buffer)
    {
        st.add(ErrorMemoryAllocationFailed);
        return st;
    }

    // Get pointers into the buffer for our three matrices
    algorithmFPType * contribsDiag = buffer + 0 * elementsInMatrix;
    algorithmFPType * contribsOff  = buffer + 1 * elementsInMatrix;
    algorithmFPType * contribsOn   = buffer + 2 * elementsInMatrix;

    // Copy nominal values (for bias term) to the condition = 0 buffer
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0ul; i < nRowsData; ++i)
    {
        contribsDiag[i * nColumnsPhi + biasTermIndex] = nominal[i];
    }

    predictContributions(iTree, nTrees, nRowsData, x, contribsDiag, 0, 0, dim);
    for (size_t i = 0ul; i < nColumnsPhi; ++i)
    {
        // initialize/reset the on/off buffers
        service_memset_seq<algorithmFPType, cpu>(contribsOff, algorithmFPType(0), 2 * elementsInMatrix);

        predictContributions(iTree, nTrees, nRowsData, x, contribsOff, -1, i, dim);
        predictContributions(iTree, nTrees, nRowsData, x, contribsOn, 1, i, dim);

        for (size_t j = 0ul; j < nRowsData; ++j)
        {
            const unsigned dataRowOffset = j * interactionMatrixSize + i * nColumnsPhi;
            const unsigned columnOffset  = j * nColumnsPhi;
            res[dataRowOffset + i]       = 0;
            for (size_t k = 0ul; k < nColumnsPhi; ++k)
            {
                // fill in the diagonal with additive effects, and off-diagonal with the interactions
                if (k == i)
                {
                    res[dataRowOffset + i] += contribsDiag[columnOffset + k];
                }
                else
                {
                    constexpr algorithmFPType half(0.5);
                    res[dataRowOffset + k] = (contribsOn[columnOffset + k] - contribsOff[columnOffset + k]) * half;
                    res[dataRowOffset + i] -= res[dataRowOffset + k];
                }
            }
        }
    }

    daal_free(buffer);

    return st;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::runInternal(services::HostAppIface * pHostApp, NumericTable * result,
                                                                          double predictionBias, bool predShapContributions,
                                                                          bool predShapInteractions)
{
    // assert we're not requesting both contributions and interactions
    DAAL_ASSERT(!(predShapContributions && predShapInteractions));

    const size_t nTreesTotal    = _aTree.size();
    const int dataNColumns      = _data->getNumberOfColumns();
    const size_t resultNColumns = result->getNumberOfColumns();
    const size_t resultNRows    = result->getNumberOfRows();

    DimType dim(*_data, nTreesTotal, getNumberOfNodes(nTreesTotal));
    WriteOnlyRows<algorithmFPType, cpu> resMatrix(result, 0, resultNRows); // select all rows for writing
    DAAL_CHECK_BLOCK_STATUS(resMatrix);
    services::internal::service_memset<algorithmFPType, cpu>(resMatrix.get(), 0, resultNRows * resultNColumns); // set nRows * nCols to 0
    SafeStatus safeStat;
    services::Status s;
    HostAppHelper host(pHostApp, 100);

    const size_t predictionIndex = resultNColumns - 1;
    for (size_t iTree = 0ul; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if (!s || host.isCancelled(s, 1)) return s;
        size_t nTreesToUse = ((iTree + dim.nTreesInBlock) < nTreesTotal ? dim.nTreesInBlock : (nTreesTotal - iTree));

        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iBlock * dim.nRowsInBlock : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);

            if (predShapContributions)
            {
                // nominal values are required to calculate the correct bias term
                TArray<algorithmFPType, cpu> nominal(nRowsToProcess);
                algorithmFPType * nominalPtr = nominal.get();
                DAAL_CHECK_MALLOC_THR(nominalPtr);
                service_memset<algorithmFPType, cpu>(nominalPtr, algorithmFPType(predictionBias), nRowsToProcess);

                // bias term: prediction - sum_i phi_i (subtraction in predictContributions)
                predict(iTree, nTreesToUse, nRowsToProcess, xBD.get(), nominalPtr, dim);

                // thread-local write rows into global result buffer
                WriteOnlyRows<algorithmFPType, cpu> resRow(result, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(resRow);

                // copy nominal predictions for bias term to correct spot in result array
                auto resRowPtr = resRow.get();
                for (size_t i = 0ul; i < nRowsToProcess; ++i)
                {
                    resRowPtr[i * resultNColumns + predictionIndex] = nominalPtr[i];
                }

                // TODO: support tree weights
                safeStat |= predictContributions(iTree, nTreesToUse, nRowsToProcess, xBD.get(), resRowPtr, 0, 0, dim);
            }
            else if (predShapInteractions)
            {
                // thread-local write rows into global result buffer
                WriteOnlyRows<algorithmFPType, cpu> resRow(result, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(resRow);

                // nominal values are required to calculate the correct bias term
                TArray<algorithmFPType, cpu> nominal(nRowsToProcess);
                algorithmFPType * nominalPtr = nominal.get();
                DAAL_CHECK_MALLOC_THR(nominalPtr);
                service_memset<algorithmFPType, cpu>(nominalPtr, algorithmFPType(predictionBias), nRowsToProcess);

                predict(iTree, nTreesToUse, nRowsToProcess, xBD.get(), nominalPtr, dim);

                // TODO: support tree weights
                safeStat |= predictContributionInteractions(iTree, nTreesToUse, nRowsToProcess, xBD.get(), nominalPtr, resRow.get(), dim);
            }
            else
            {
                algorithmFPType * res = resMatrix.get() + iStartRow;
                if ((predictionBias < 0 && predictionBias < -DBL_EPSILON) || (0 < predictionBias && DBL_EPSILON < predictionBias))
                {
                    // memory is already initialized to 0
                    // only set it to the bias term if it's != 0
                    service_memset<algorithmFPType, cpu>(res, algorithmFPType(predictionBias), nRowsToProcess);
                }
                predict(iTree, nTreesToUse, nRowsToProcess, xBD.get(), res, dim);
            }
        });

        s = safeStat.detach();
    }

    return s;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
algorithmFPType PredictRegressionTask<algorithmFPType, cpu>::predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                            const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType val = 0;
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
        val += gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*_aTree[iTree], _featHelper, x, dispatcher);
    return val;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
void PredictRegressionTask<algorithmFPType, cpu>::predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                       algorithmFPType * res,
                                                                       const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType v[vectorBlockSize];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu, hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(
            *_aTree[iTree], _featHelper, x, v, dispatcher);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t row = 0ul; row < vectorBlockSize; ++row)
        {
            res[row] += v[row];
        }
    }
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
