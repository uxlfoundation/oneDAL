/* file: multiclassclassifier_predict_votebased_impl.i */
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
//  Implementation of score-based method for Multi-class classifier
//  prediction algorithm.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_VOTEBASED_IMPL_I__
#define __MULTICLASSCLASSIFIER_PREDICT_VOTEBASED_IMPL_I__

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel<voteBased, training::oneAgainstOne, algorithmFPType, cpu> : public Kernel
{
    Status compute(const NumericTable * a, const daal::algorithms::Model * m, SvmModel * svmModel, NumericTable * pred, NumericTable * df,
                   const daal::algorithms::Parameter * par);
};

/** Base class for threading subtask */
template <typename algorithmFPType, CpuType cpu>
class SubTaskVoteBased
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubTaskVoteBased() {}

    /**
     * Computes a block of predictions
     * \param[in] startRow  Index of the starting row in the block
     * \param[in] nRows     Number of rows in the block
     * \param[in] a         Numeric table of size n x p with input data set
     * \param[in] model     Model of the multi-class classifier
     * \param[out] r        Numeric table of size n x 1 with resulting labels
     * \param[in] nonEmptyClassMap Array that contains indices of non-empty classes
     * \return Status of the computations
     */
    Status predict(size_t startRow, size_t nRows, const NumericTable * a, Model * model, NumericTable * pred, NumericTable * df,
                   const size_t * nonEmptyClassMap, const size_t * classIndicesData, const bool isSvmModel)
    {
        Status s;
        int * votes          = _aVotes.get(); // array of two-class classifiers' votes
        algorithmFPType * y  = _aY.get();     // array of two-class classifier predictions
        const size_t nModels = (_nClasses * (_nClasses - 1)) >> 1;
        daal::services::internal::service_memset_seq<int, cpu>(votes, 0, _nClasses * nRows);
        NumericTablePtr xTable; // block of rows from the input data set
        s = getDataBlock(startRow, nRows, a, xTable);
        if (!s) return s;

        if (!df && nRows != _yTable->getNumberOfRows()) _yTable->resize(nRows);

        /* TODO: Try threading here */
        for (size_t imodel = 0; imodel < nModels; ++imodel)
        {
            const size_t iClass = classIndicesData[imodel];
            const size_t jClass = classIndicesData[imodel + nModels];
            /* Compute two-class predictions for the pair of classes (iClass, jClass)
                       for the block of input observations */
            classifier::prediction::Input * input = _simplePrediction->getInput();
            DAAL_CHECK(input, ErrorNullInput);
            input->set(classifier::prediction::data, xTable);
            input->set(classifier::prediction::model, model->getTwoClassClassifierModel(imodel));

            const size_t iClassesForDF = isSvmModel ? imodel : (jClass * (2 * _nClasses - jClass - 1)) / 2 + (iClass - jClass - 1);
            WriteOnlyColumns<algorithmFPType, cpu> dfBlock(df, iClassesForDF, startRow, nRows);
            DAAL_CHECK_BLOCK_STATUS(dfBlock);
            algorithmFPType * dfData = dfBlock.get();

            if (dfData)
            {
                y       = dfData;
                _yTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(dfData, 1, nRows);
                _yRes->set(classifier::prediction::prediction, _yTable);
                _simplePrediction->setResult(_yRes);
            }

            s = _simplePrediction->computeNoThrow();
            if (!s) return Status(ErrorMultiClassFailedToComputeTwoClassPrediction).add(s);

            /* Compute votes for the block of input observations */
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < nRows; ++i)
            {
                if (y[i] >= 0)
                    votes[i * _nClasses + iClass]++;
                else
                    votes[i * _nClasses + jClass]++;
            }
        }

        if (pred)
        {
            /* Compute resulting labels as indices of the maximum vote values */
            WriteOnlyRows<int, cpu> res(pred, startRow, nRows);
            int * labels = res.get();
            DAAL_CHECK_MALLOC(labels);

            int * votesPtr = votes;
            for (size_t i = 0; i < nRows; i++, votesPtr += _nClasses)
            {
                labels[i]   = nonEmptyClassMap[0];
                int maxVote = votesPtr[0];
                for (size_t iClass = 1; iClass < _nClasses; iClass++)
                {
                    if (votesPtr[iClass] > maxVote)
                    {
                        maxVote   = votesPtr[iClass];
                        labels[i] = nonEmptyClassMap[iClass];
                    }
                }
            }
        }
        return s;
    }

protected:
    /**
     * Get block of input observations
     * \param[in] startRow  Index of the starting row in the block
     * \param[in] nRows     Number of rows in the block
     * \param[in] a         Numeric table of size n x p with input data set
     * \param[out] xTable   Numeric table of size nRows x p with the block of observations
     * \return Status of the computations
     */
    virtual Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) = 0;

    /**
     * Constructs a base threading subtask
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \param[out] valid            Flag. True if the task was constructed successfully, false otherwise
     */
    SubTaskVoteBased(size_t nClasses, size_t nRows, const SharedPtr<classifier::prediction::Batch> & simplePrediction, bool isComputeDecisionFunction,
                     bool & valid)
        : _nClasses(nClasses),
          _aVotes(nClasses * nRows),
          _yRes(new typename classifier::prediction::Batch::ResultType()),
          _simplePrediction(simplePrediction->clone())
    {
        if (!_aVotes.get() || !_yRes)
        {
            valid = false;
            return;
        }

        if (!isComputeDecisionFunction)
        {
            _aY.reset(nRows);
            if (!_aY.get())
            {
                valid = false;
                return;
            }
            _yTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_aY.get(), 1, nRows);
            if (!_yTable)
            {
                valid = false;
                return;
            }
            _yRes->set(classifier::prediction::prediction, _yTable);
            _simplePrediction->setResult(_yRes);
        }
    }

private:
    size_t _nClasses;
    TArray<algorithmFPType, cpu> _aY;
    TArray<int, cpu> _aVotes;
    NumericTablePtr _yTable;
    services::SharedPtr<typename classifier::prediction::Batch::ResultType> _yRes;
    SharedPtr<classifier::prediction::Batch> _simplePrediction;
}; // namespace internal

/** Class for threading subtask that works with dense input data */
template <typename algorithmFPType, CpuType cpu>
class SubTaskVoteBasedDense : public SubTaskVoteBased<algorithmFPType, cpu>
{
public:
    typedef SubTaskVoteBased<algorithmFPType, cpu> super;
    using super::predict;

    /**
     * Constructs a threading subtask that works with dense input data
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \return Pointer to the newly constructed subtask in case of success; NULL pointer in case of failure
     */
    static super * create(size_t nClasses, size_t nRows, const SharedPtr<classifier::prediction::Batch> & simplePrediction,
                          bool isComputeDecisionFunction)
    {
        bool valid  = true;
        super * res = new SubTaskVoteBasedDense(nClasses, nRows, simplePrediction, isComputeDecisionFunction, valid);
        if (res && valid) return res;
        delete res;
        return nullptr;
    }

protected:
    Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) DAAL_C11_OVERRIDE
    {
        _xRows.set(const_cast<NumericTable *>(a), startRow, nRows);
        DAAL_CHECK_BLOCK_STATUS(_xRows);
        Status s;
        xTable =
            HomogenNumericTableCPU<algorithmFPType, cpu>::create(const_cast<algorithmFPType *>(_xRows.get()), a->getNumberOfColumns(), nRows, &s);
        return s;
    }

private:
    SubTaskVoteBasedDense(size_t nClasses, size_t nRows, const SharedPtr<classifier::prediction::Batch> & simplePrediction,
                          bool isComputeDecisionFunction, bool & valid)
        : super(nClasses, nRows, simplePrediction, isComputeDecisionFunction, valid)
    {}
    ReadRows<algorithmFPType, cpu> _xRows;
};

/** Class for threading subtask that works with CSR input data */
template <typename algorithmFPType, CpuType cpu>
class SubTaskVoteBasedCSR : public SubTaskVoteBased<algorithmFPType, cpu>
{
public:
    typedef SubTaskVoteBased<algorithmFPType, cpu> super;
    using super::predict;

    /**
     * Constructs a threading subtask that works with CSR input data
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \return Pointer to the newly constructed subtask in case of success; NULL pointer in case of failure
     */
    static super * create(size_t nClasses, size_t nRows, const SharedPtr<classifier::prediction::Batch> & simplePrediction,
                          bool isComputeDecisionFunction)
    {
        bool valid  = true;
        super * res = new SubTaskVoteBasedCSR(nClasses, nRows, simplePrediction, isComputeDecisionFunction, valid);
        if (res && valid) return res;
        delete res;
        res = nullptr;
        return nullptr;
    }

protected:
    Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) DAAL_C11_OVERRIDE
    {
        const bool toOneBaseRowIndices = true;
        _xRows.set(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a)), startRow, nRows, toOneBaseRowIndices);
        DAAL_CHECK_BLOCK_STATUS(_xRows);

        algorithmFPType * const values = const_cast<algorithmFPType *>(_xRows.values());
        size_t * const cols            = const_cast<size_t *>(_xRows.cols());
        size_t * const rows            = const_cast<size_t *>(_xRows.rows());
        Status s;
        xTable = CSRNumericTable::create(values, cols, rows, a->getNumberOfColumns(), nRows, CSRNumericTableIface::oneBased, &s);
        return s;
    }

private:
    SubTaskVoteBasedCSR(size_t nClasses, size_t nRows, const SharedPtr<classifier::prediction::Batch> & simplePrediction,
                        bool isComputeDecisionFunction, bool & valid)
        : super(nClasses, nRows, simplePrediction, isComputeDecisionFunction, valid)
    {}

    ReadRowsCSR<algorithmFPType, cpu> _xRows;
};

template <typename algorithmFPType, CpuType cpu>
Status MultiClassClassifierPredictKernel<voteBased, training::oneAgainstOne, algorithmFPType, cpu>::compute(const NumericTable * a,
                                                                                                            const daal::algorithms::Model * m,
                                                                                                            SvmModel * svmModel, NumericTable * pred,
                                                                                                            NumericTable * df,
                                                                                                            const daal::algorithms::Parameter * par)
{
    Model * model                              = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
    multi_class_classifier::Parameter * mccPar = static_cast<multi_class_classifier::Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nClasses                            = mccPar->nClasses;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClasses, sizeof(size_t));

    TArray<size_t, cpu> nonEmptyClassMapBuffer(nClasses);
    DAAL_CHECK_MALLOC(nonEmptyClassMapBuffer.get());

    const bool isSvmModel = svmModel != nullptr;

    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nModels, 2);
    TArray<size_t, cpu> classIndices(nModels * 2);
    DAAL_CHECK_MALLOC(classIndices.get());
    size_t * classIndicesData = classIndices.get();
    Status s                  = getClassIndices<algorithmFPType, cpu>(nClasses, isSvmModel, classIndicesData);
    DAAL_CHECK_STATUS_VAR(s);

    size_t * nonEmptyClassMap = (size_t *)nonEmptyClassMapBuffer.get();
    s |= getNonEmptyClassMap<algorithmFPType, cpu>(nClasses, model, classIndicesData, nonEmptyClassMap);
    DAAL_CHECK_STATUS_VAR(s);

    const size_t nVectors = a->getNumberOfRows();

    SharedPtr<classifier::prediction::Batch> simplePrediction = mccPar->prediction;

    const size_t nRowsInBlock = 256;
    size_t nBlocks            = nVectors / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nVectors) nBlocks++;

    typedef SubTaskVoteBased<algorithmFPType, cpu> TSubTask;
    daal::ls<TSubTask *> lsTask([=, &simplePrediction]() {
        if (a->getDataLayout() == NumericTableIface::csrArray)
            return SubTaskVoteBasedCSR<algorithmFPType, cpu>::create(nClasses, nRowsInBlock, simplePrediction, df);
        return SubTaskVoteBasedDense<algorithmFPType, cpu>::create(nClasses, nRowsInBlock, simplePrediction, df);
    });

    /* Process input data set block by block */
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        TSubTask * local = lsTask.local();
        if (!local)
        {
            safeStat.add(ErrorMemoryAllocationFailed);
            return;
        }
        DAAL_LS_RELEASE(TSubTask, lsTask, local); //releases local storage when leaving this scope

        const size_t startRow = iBlock * nRowsInBlock;
        const size_t nRows    = (startRow + nRowsInBlock > nVectors) ? nVectors - startRow : nRowsInBlock;

        /* Get a block of predictions */
        Status s = local->predict(startRow, nRows, a, model, pred, df, nonEmptyClassMap, classIndicesData, isSvmModel);
        DAAL_CHECK_STATUS_THR(s);
    });

    lsTask.reduce([=, &safeStat](TSubTask * local) { delete local; });
    return safeStat.detach();
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
