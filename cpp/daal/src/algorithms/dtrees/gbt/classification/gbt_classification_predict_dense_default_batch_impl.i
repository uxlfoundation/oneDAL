/* file: gbt_classification_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees classification predictions calculation
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_predict_kernel.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_dense_default_batch_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include "src/algorithms/objective_function/cross_entropy_loss/cross_entropy_loss_dense_default_batch_kernel.h"
#include "src/services/service_algo_utils.h"
#include <cfloat>

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask - declaration
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictBinaryClassificationTask : public gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu>
{
public:
    typedef gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu> super;

public:
    /**
     * \brief Construct a new Predict Binary Classification Task object
     *
     * \param x NumericTable observation data
     * \param y NumericTable prediction data
     * \param prob NumericTable probability data
     */
    PredictBinaryClassificationTask(const NumericTable * x, NumericTable * y, NumericTable * prob) : super(x, y), _prob(prob) {}

    /**
     * \brief Run prediction for the given model
     *
     * \param m The model for which to run prediction
     * \param nIterations Number of iterations
     * \param pHostApp HostAppInterface
     * \param predShapContributions Predict SHAP contributions
     * \param predShapInteractions Predict SHAP interactions
     * \return services::Status
     */
    services::Status run(const gbt::classification::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp,
                         bool predShapContributions, bool predShapInteractions);

protected:
    /**
     * \brief Convert the model bias to a margin, considering the softmax activation
     *
     * \param bias Bias in class score units
     * \return algorithmFPType Bias in softmax offset units
     */
    algorithmFPType getMarginFromModelBias(algorithmFPType bias) const;

protected:
    NumericTable * _prob;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictMulticlassTask - declaration
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictMulticlassTask
{
public:
    typedef gbt::internal::GbtDecisionTree TreeType;
    typedef gbt::prediction::internal::TileDimensions<algorithmFPType> DimType;
    typedef daal::tls<algorithmFPType *> ClassesRawBoostedTlsBase;
    typedef daal::TlsMem<algorithmFPType, cpu> ClassesRawBoostedTls;

    /**
     * \brief Construct a new Predict Multiclass Task object
     *
     * \param x NumericTable observation data
     * \param y NumericTable prediction data
     * \param prob NumericTable probability data
     */
    PredictMulticlassTask(const NumericTable * x, NumericTable * y, NumericTable * prob) : _data(x), _res(y), _prob(prob) {}

    /**
     * \brief Run prediction for the given model
     *
     * \param m The model for which to run prediction
     * \param nClasses Number of data classes
     * \param nIterations Number of iterations
     * \param pHostApp HostAppInterface
     * \param predShapContributions Predict SHAP contributions
     * \param predShapInteractions Predict SHAP interactions
     * \return services::Status
     */
    services::Status run(const gbt::classification::internal::ModelImpl * m, size_t nClasses, size_t nIterations, services::HostAppIface * pHostApp,
                         bool predShapContributions, bool predShapInteractions);

protected:
    /** Dispatcher type for template dispatching */
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    using dispatcher_t = gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing>;

    /**
     * \brief Helper boolean constant to populate template dispatcher
     *
     * \param val A boolean value, known at compile time
     */
    template <bool val>
    struct BooleanConstant
    {
        typedef BooleanConstant<val> type;
    };

    /**
     * \brief Run prediction for all trees
     *
     * \param nTreesTotal Total number of trees in model
     * \param nClasses Number of data classes
     * \param bias Global prediction bias (e.g. base_score in XGBoost)
     * \param dim DimType helper
     * \return services::Status
     */
    services::Status predictByAllTrees(size_t nTreesTotal, size_t nClasses, double bias, const DimType & dim);

    /**
     * \brief Make prediction for a number of trees
     *
     * \param hasUnorderedFeatures Data has unordered features yes/no
     * \param hasAnyMissing Data has missing values yes/no
     * \param val Output prediction
     * \param iFirstTree Index of first ree
     * \param nTrees Number of trees included in prediction
     * \param nClasses Number of data classes
     * \param x Input observation data
     * \param dispatcher Template dispatcher helper
     */
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    void predictByTrees(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType * x,
                        const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);

    /**
     * \brief Make prediction for a number of trees leveraging vector instructions
     *
     * \param hasUnorderedFeatures Data has unordered features yes/no
     * \param hasAnyMissing Data has missing values yes/no
     * \param vectorBlockSize Vector instruction block size
     * \param val Output prediction
     * \param iFirstTree Index of first ree
     * \param nTrees Number of trees included in prediction
     * \param nClasses Number of data classes
     * \param x Input observation data
     * \param dispatcher Template dispatcher helper
     */
    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    void predictByTreesVector(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType * x,
                              const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);

    /**
     * \brief Assign a class index to the result
     *
     * \param res Pointer to result array
     * \param val Value of current prediction
     * \param iRow
     * \param i
     * \param nClasses Number of data classes
     * \param dispatcher Template dispatcher helper
     */
    inline void updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses, BooleanConstant<true> dispatcher);

    /**
     * \brief Empty function if results assigning is not required.
     *
     * \param res Pointer to result array
     * \param val Value of current prediction
     * \param iRow
     * \param i
     * \param nClasses Number of data classes
     * \param dispatcher Template dispatcher helper
     */
    inline void updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses, BooleanConstant<false> dispatcher);

    /**
     * \brief Prepare buff pointer for the next using. All steps reuse the same memory.
     *
     * \param buff Pointer to a buffer
     * \param buf_shift
     * \param buf_size
     * \param dispatcher
     * \return algorithmFPType* Pointer to the input buffer
     */
    inline algorithmFPType * updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size, BooleanConstant<true> dispatcher);

    /**
     * \brief Prepare buff pointer for the next using. Steps have own memory.
     *
     * \param buff
     * \param buf_shift
     * \param buf_size
     * \param dispatcher
     * \return algorithmFPType*
     */
    inline algorithmFPType * updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size, BooleanConstant<false> dispatcher);

    /**
     * \brief Get the total number of nodes in all trees for tree number [1, 2, ... nTrees]
     *
     * \param nTrees Number of trees that contribute to the sum
     * \return size_t Number of nodes in all contributing trees
     */
    inline size_t getNumberOfNodes(size_t nTrees);

    /**
     * \brief Check for missing data
     *
     * \param x Input observation data
     * \param nTrees Number of contributing trees
     * \param nRows Number of rows in input observation data to be considered
     * \param nColumns Number of columns in input observation data to be considered
     * \return true If runtime check for missing is required
     * \return false If runtime check for missing is not required
     */
    inline bool checkForMissing(const algorithmFPType * x, size_t nTrees, size_t nRows, size_t nColumns);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param hasUnorderedFeatures Data has unordered features yes/no
     * \param hasAnyMissing Data has missing values yes/no
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param vectorBlockSize Vector instruction block size
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     */
    template <bool hasUnorderedFeatures, bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param hasAnyMissing Data has missing values yes/no
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param vectorBlockSize Vector instruction block size
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     */
    template <bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param vectorBlockSize Vector instruction block size
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     */
    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param vectorBlockSizeFactor Vector instruction block size - recursively decremented until it becomes equal to dim.vectorBlockSizeFactor or equal to DimType::minVectorBlockSizeFactor
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     */
    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim, BooleanConstant<true> keepLooking);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param vectorBlockSizeFactor Vector instruction block size - recursively decremented until it becomes equal to dim.vectorBlockSizeFactor or equal to DimType::minVectorBlockSizeFactor
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     * \param dim DimType helper
     * \param keepLooking
     */
    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim, BooleanConstant<false> keepLooking);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param isResValidPtr Result pointer is valid yes/no (write result to the pointer if yes, skip if no)
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     * \param dim DimType helper
     */
    template <bool isResValidPtr, bool reuseBuffer>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim);

    /**
     * \brief Traverse a number of trees to get prediction results
     *
     * \param reuseBuffer Re-use buffer yes/no (will fill buffer with zero if yes, shift buff pointer if no)
     * \param nTrees Number of trees contributing to prediction
     * \param nClasses Number of data classes
     * \param nRows Number of rows in observation data for which prediction is run
     * \param nColumns Number of columns in observation data
     * \param x Input observation data
     * \param buff A pre-allocated buffer for computations
     * \param[out] res Output prediction result
     * \param dim DimType helper
     */
    template <bool reuseBuffer>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim);

    /**
     * \brief Get index of element with maximum value / activation
     *
     * \param val Pointer to values
     * \param nClasses Number of columns per value
     * \return size_t The column index with the maximum value
     */
    size_t getMaxClass(const algorithmFPType * val, size_t nClasses) const;

protected:
    const NumericTable * _data;
    NumericTable * _res;
    NumericTable * _prob;
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const TreeType *, cpu> _aTree;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask - implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
services::Status PredictBinaryClassificationTask<algorithmFPType, cpu>::run(const gbt::classification::internal::ModelImpl * m, size_t nIterations,
                                                                            services::HostAppIface * pHostApp, bool predShapContributions,
                                                                            bool predShapInteractions)
{
    // assert we're not requesting both contributions and interactions
    DAAL_ASSERT(!(predShapContributions && predShapInteractions));

    DAAL_ASSERT(!nIterations || nIterations <= m->size());
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    const auto nTreesTotal = (nIterations ? nIterations : m->size());
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);

    const auto nRows = this->_data->getNumberOfRows();
    services::Status s;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(algorithmFPType));

    // we convert the bias to a margin if it's > 0
    // otherwise the margin is 0
    algorithmFPType margin(0);
    if (m->getPredictionBias() > FLT_EPSILON)
    {
        margin = getMarginFromModelBias(m->getPredictionBias());
    }

    // compute raw boosted values
    if (this->_res && _prob)
    {
        WriteOnlyRows<algorithmFPType, cpu> resBD(this->_res, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(resBD);
        const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
        algorithmFPType * res          = resBD.get();
        WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(probBD);
        algorithmFPType * prob_pred = probBD.get();
        TArray<algorithmFPType, cpu> expValPtr(nRows);
        algorithmFPType * expVal = expValPtr.get();
        DAAL_CHECK_MALLOC(expVal);
        s = super::runInternal(pHostApp, this->_res, margin, predShapContributions, predShapInteractions);
        if (!s) return s;

        auto nBlocks           = daal::threader_get_threads_number();
        const size_t blockSize = nRows / nBlocks;
        nBlocks += (nBlocks * blockSize != nRows);

        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow  = iBlock * blockSize;
            const size_t finishRow = (((iBlock + 1) == nBlocks) ? nRows : (iBlock + 1) * blockSize);
            daal::internal::MathInst<algorithmFPType, cpu>::vExp(finishRow - startRow, res + startRow, expVal + startRow);

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iRow = startRow; iRow < finishRow; ++iRow)
            {
                res[iRow]               = label[services::internal::SignBit<algorithmFPType, cpu>::get(res[iRow])];
                prob_pred[2 * iRow + 1] = expVal[iRow] / (algorithmFPType(1.) + expVal[iRow]);
                prob_pred[2 * iRow]     = algorithmFPType(1.) - prob_pred[2 * iRow + 1];
            }
        });
    }

    else if ((!this->_res) && _prob)
    {
        WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(probBD);
        algorithmFPType * prob_pred = probBD.get();
        TArray<algorithmFPType, cpu> expValPtr(nRows);
        algorithmFPType * expVal = expValPtr.get();
        NumericTablePtr expNT    = HomogenNumericTableCPU<algorithmFPType, cpu>::create(expVal, 1, nRows, &s);
        DAAL_CHECK_MALLOC(expVal);
        s = super::runInternal(pHostApp, expNT.get(), margin, predShapContributions, predShapInteractions);
        if (!s) return s;

        auto nBlocks           = daal::threader_get_threads_number();
        const size_t blockSize = nRows / nBlocks;
        nBlocks += (nBlocks * blockSize != nRows);
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow  = iBlock * blockSize;
            const size_t finishRow = (((iBlock + 1) == nBlocks) ? nRows : (iBlock + 1) * blockSize);
            daal::internal::MathInst<algorithmFPType, cpu>::vExp(finishRow - startRow, expVal + startRow, expVal + startRow);
            for (size_t iRow = startRow; iRow < finishRow; ++iRow)
            {
                prob_pred[2 * iRow + 1] = expVal[iRow] / (algorithmFPType(1.) + expVal[iRow]);
                prob_pred[2 * iRow]     = algorithmFPType(1.) - prob_pred[2 * iRow + 1];
            }
        });
    }
    else if (this->_res && (!_prob))
    {
        WriteOnlyRows<algorithmFPType, cpu> resBD(this->_res, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(resBD);
        const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
        algorithmFPType * res          = resBD.get();
        s                              = super::runInternal(pHostApp, this->_res, margin, predShapContributions, predShapInteractions);
        if (!s) return s;

        // for SHAP values, the score from runInternal is what we need
        if (!(predShapContributions || predShapInteractions))
        {
            // convert the score to a class label
            typedef services::internal::SignBit<algorithmFPType, cpu> SignBit;
            PRAGMA_FORCE_SIMD
            for (size_t iRow = 0; iRow < nRows; ++iRow)
            {
                // probability is a sigmoid(f) hence sign(f) can be checked
                const algorithmFPType initial = res[iRow];
                const int sign                = SignBit::get(initial);
                res[iRow]                     = label[sign];
            }
        }
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
algorithmFPType PredictBinaryClassificationTask<algorithmFPType, cpu>::getMarginFromModelBias(algorithmFPType bias) const
{
    DAAL_ASSERT((0.0 < bias) && (bias < 1.0));
    constexpr algorithmFPType one(1);
    // convert bias to margin
    return -one * daal::internal::MathInst<algorithmFPType, cpu>::sLog(one / bias - one);
}

//////////////////////////////////////////////////////////////////////////////////////////
// PredictMulticlassTask - implementation
//////////////////////////////////////////////////////////////////////////////////////////

template <typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses,
                                                               BooleanConstant<true> dispatcher)
{
    res[iRow + i] = getMaxClass(val + i * nClasses, nClasses);
}

template <typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses,
                                                               BooleanConstant<false> dispatcher)
{}

template <typename algorithmFPType, CpuType cpu>
algorithmFPType * PredictMulticlassTask<algorithmFPType, cpu>::updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size,
                                                                            BooleanConstant<true> dispatcher)
{
    services::internal::service_memset_seq<algorithmFPType, cpu>(buff, algorithmFPType(0), buf_size);
    return buff;
}

template <typename algorithmFPType, CpuType cpu>
algorithmFPType * PredictMulticlassTask<algorithmFPType, cpu>::updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size,
                                                                            BooleanConstant<false> dispatcher)
{
    return buff + buf_shift;
}

template <typename algorithmFPType, CpuType cpu>
size_t PredictMulticlassTask<algorithmFPType, cpu>::getNumberOfNodes(size_t nTrees)
{
    size_t nNodesTotal = 0;
    for (size_t iTree = 0; iTree < nTrees; ++iTree)
    {
        nNodesTotal += this->_aTree[iTree]->getNumberOfNodes();
    }
    return nNodesTotal;
}

template <typename algorithmFPType, CpuType cpu>
bool PredictMulticlassTask<algorithmFPType, cpu>::checkForMissing(const algorithmFPType * x, size_t nTrees, size_t nRows, size_t nColumns)
{
    size_t nLvlTotal = 0;
    for (size_t iTree = 0; iTree < nTrees; ++iTree)
    {
        nLvlTotal += this->_aTree[iTree]->getMaxLvl();
    }
    if (nLvlTotal <= nColumns)
    {
        // Checking is complicated. Better to do it during inference.
        return true;
    }
    else
    {
        for (size_t idx = 0; idx < nRows * nColumns; ++idx)
        {
            if (checkFinitenessByComparison(x[idx])) return true;
        }
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res)
{
    dispatcher_t<hasUnorderedFeatures, hasAnyMissing> dispatcher;
    size_t iRow = 0;
    for (; iRow + vectorBlockSize <= nRows; iRow += vectorBlockSize)
    {
        algorithmFPType * val = updateBuffer(buff, iRow * nClasses, nClasses * vectorBlockSize, BooleanConstant<reuseBuffer>());
        predictByTreesVector<hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(val, 0, nTrees, nClasses, x + iRow * nColumns, dispatcher);
        for (size_t i = 0; i < vectorBlockSize; ++i)
        {
            updateResult(res, val, iRow, i, nClasses, BooleanConstant<isResValidPtr>());
        }
    }
    for (; iRow < nRows; ++iRow)
    {
        algorithmFPType * val = updateBuffer(buff, iRow * nClasses, nClasses, BooleanConstant<reuseBuffer>());
        predictByTrees(val, 0, nTrees, nClasses, x + iRow * nColumns, dispatcher);
        updateResult(res, val, iRow, 0, nClasses, BooleanConstant<isResValidPtr>());
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res)
{
    if (this->_featHelper.hasUnorderedFeatures())
    {
        predict<true, hasAnyMissing, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }
    else
    {
        predict<false, hasAnyMissing, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res)
{
    const bool hasAnyMissing = checkForMissing(x, nTrees, nRows, nColumns);
    if (hasAnyMissing)
    {
        predict<true, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }
    else
    {
        predict<false, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res, const DimType & dim,
                                                          BooleanConstant<true> keepLooking)
{
    constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
    if (dim.vectorBlockSizeFactor == vectorBlockSizeFactor)
    {
        predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor * vectorBlockSizeStep>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }
    else
    {
        predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor - 1>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim,
                                                                       BooleanConstant<vectorBlockSizeFactor != DimType::minVectorBlockSizeFactor>());
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res, const DimType & dim,
                                                          BooleanConstant<false> keepLooking)
{
    constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
    predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor * vectorBlockSizeStep>(nTrees, nClasses, nRows, nColumns, x, buff, res);
}

template <typename algorithmFPType, CpuType cpu>
template <bool isResValidPtr, bool reuseBuffer>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res, const DimType & dim)
{
    constexpr size_t maxVectorBlockSizeFactor = DimType::maxVectorBlockSizeFactor;
    if (maxVectorBlockSizeFactor > 1)
    {
        predict<isResValidPtr, reuseBuffer, maxVectorBlockSizeFactor>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim, BooleanConstant<true>());
    }
    else
    {
        predict<isResValidPtr, reuseBuffer, maxVectorBlockSizeFactor>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim, BooleanConstant<false>());
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool reuseBuffer>
void PredictMulticlassTask<algorithmFPType, cpu>::predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x,
                                                          algorithmFPType * buff, algorithmFPType * res, const DimType & dim)
{
    if (res)
    {
        predict<true, reuseBuffer>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim);
    }
    else
    {
        predict<false, reuseBuffer>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim);
    }
}

template <typename algorithmFPType, CpuType cpu>
size_t PredictMulticlassTask<algorithmFPType, cpu>::getMaxClass(const algorithmFPType * val, size_t nClasses) const
{
    return services::internal::getMaxElementIndex<algorithmFPType, cpu>(val, nClasses);
}

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const classification::Model * m, NumericTable * r, NumericTable * prob,
                                                                      size_t nClasses, size_t nIterations, bool predShapContributions,
                                                                      bool predShapInteractions)
{
    const daal::algorithms::gbt::classification::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl *>(m);
    if (nClasses == 2)
    {
        PredictBinaryClassificationTask<algorithmFPType, cpu> task(x, r, prob);
        return task.run(pModel, nIterations, pHostApp, predShapContributions, predShapInteractions);
    }
    PredictMulticlassTask<algorithmFPType, cpu> task(x, r, prob);
    return task.run(pModel, nClasses, nIterations, pHostApp, predShapContributions, predShapInteractions);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::run(const gbt::classification::internal::ModelImpl * m, size_t nClasses,
                                                                  size_t nIterations, services::HostAppIface * pHostApp, bool predShapContributions,
                                                                  bool predShapInteractions)
{
    // assert we're not requesting both contributions and interactions
    DAAL_ASSERT(!(predShapContributions && predShapInteractions));

    DAAL_ASSERT(!nIterations || nClasses * nIterations <= m->size());
    const auto nTreesTotal = (nIterations ? nIterations * nClasses : m->size());
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);

    DimType dim(*_data, nTreesTotal, getNumberOfNodes(nTreesTotal));

    return predictByAllTrees(nTreesTotal, nClasses, m->getPredictionBias(), dim);
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTrees(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses,
                                                                 const algorithmFPType * x,
                                                                 const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        val[iTree % nClasses] +=
            gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x, dispatcher);
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTreesVector(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses,
                                                                       const algorithmFPType * x,
                                                                       const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType v[vectorBlockSize];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu, hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(
            *this->_aTree[iTree], this->_featHelper, x, v, dispatcher);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < vectorBlockSize; ++j) val[(iTree % nClasses) + j * nClasses] += v[j];
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::predictByAllTrees(size_t nTreesTotal, size_t nClasses, double bias, const DimType & dim)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, dim.nRowsTotal);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    const size_t nCols(_data->getNumberOfColumns());
    const size_t nRows(_data->getNumberOfRows());
    daal::SafeStatus safeStat;
    if (_prob)
    {
        WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, dim.nRowsTotal);
        DAAL_CHECK_BLOCK_STATUS(probBD);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nClasses);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows * nClasses, sizeof(algorithmFPType));
        TArray<algorithmFPType, cpu> valPtr(nRows * nClasses);
        algorithmFPType * valFull = valPtr.get();
        services::internal::service_memset<algorithmFPType, cpu>(valFull, algorithmFPType(bias), nRows * nClasses);

        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == (dim.nDataBlocks - 1)) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
            algorithmFPType * valL      = valFull + iStartRow * nClasses;
            algorithmFPType * val       = valL;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() ? resBD.get() + iStartRow : nullptr;

            predict<false>(nTreesTotal, nClasses, nRowsToProcess, nCols, xBD.get(), valL, res, dim);
        });
        algorithmFPType * prob_pred = probBD.get();
        daal::algorithms::optimization_solver::cross_entropy_loss::internal::CrossEntropyLossKernel<
            algorithmFPType, daal::algorithms::optimization_solver::cross_entropy_loss::defaultDense, cpu>::softmaxThreaded(valFull, prob_pred, nRows,
                                                                                                                            nClasses);
    }
    else if (!_prob && this->_res)
    {
        ClassesRawBoostedTls lsData(nClasses * dim.vectorBlockSizeFactor * dim.vectorBlockSizeStep);
        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            algorithmFPType * const val = lsData.local();
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == (dim.nDataBlocks - 1)) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() + iStartRow;

            predict<true, true>(nTreesTotal, nClasses, nRowsToProcess, nCols, xBD.get(), val, res, dim);
        });
    }

    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
