/* file: multiclassclassifier_predict_mccwu_impl.i */
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
//  Implementation of Wu method for Multi-class classifier
//  prediction algorithm.
//--
*/
/*
//  REFERENCES
//
//  1. Ting-Fan Wu, Chih-Jen Lin, Ruby C. Weng
//     Probability Estimates for Multi-class Classification by Pairwise Coupling,
//     Journal of Machine Learning Research 5, 2004.
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_MCCWU_IMPL_I__
#define __MULTICLASSCLASSIFIER_PREDICT_MCCWU_IMPL_I__

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "src/threading/threading.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;

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
template <typename algorithmFPType, CpuType cpu>
services::Status MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>::compute(
    const NumericTable * a, const daal::algorithms::Model * m, svm::internal::ModelImpl * svmModel, NumericTable * pred, NumericTable * df,
    const daal::algorithms::Parameter * par)
{
    Model * model                              = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
    multi_class_classifier::Parameter * mccPar = static_cast<multi_class_classifier::Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nClasses                            = mccPar->nClasses;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClasses, sizeof(size_t));

    TArray<size_t, cpu> nonEmptyClassMapBuffer(nClasses);
    DAAL_CHECK_MALLOC(nonEmptyClassMapBuffer.get());

    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nModels, 2);
    TArray<size_t, cpu> classIndices(nModels * 2);
    DAAL_CHECK_MALLOC(classIndices.get());
    services::Status s = getClassIndices<algorithmFPType, cpu>(nClasses, svmModel != nullptr, classIndices.get());
    DAAL_CHECK_STATUS_VAR(s);

    size_t * nonEmptyClassMap = (size_t *)nonEmptyClassMapBuffer.get();
    s |= getNonEmptyClassMap<algorithmFPType, cpu>(nClasses, model, classIndices.get(), nonEmptyClassMap);
    DAAL_CHECK_STATUS_VAR(s);

    const size_t nIter = mccPar->maxIterations;
    const double eps   = mccPar->accuracyThreshold;

    const size_t nFeatures = a->getNumberOfColumns();
    const size_t nVectors  = a->getNumberOfRows();

    size_t nRowsInBlock = getMultiClassClassifierPredictBlockSize<algorithmFPType, cpu>();
    /* Calculate number of blocks of rows including tail block */
    size_t nBlocks = nVectors / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nVectors)
    {
        nBlocks++;
    }

    /* Allocate data for storing of intermediate results and subsets of input data*/
    typedef SubTask<algorithmFPType, cpu> TSubTask;
    daal::ls<TSubTask *> lsTask([=]() {
        if (a->getDataLayout() == NumericTableIface::csrArray)
            return (TSubTask *)SubTaskCSR<algorithmFPType, cpu>::create(nClasses, nRowsInBlock, a, df, mccPar->prediction);
        return (TSubTask *)SubTaskDense<algorithmFPType, cpu>::create(nClasses, nRowsInBlock, a, df, mccPar->prediction);
    });

    daal::SafeStatus safeStat;
    daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
        const size_t startRow = iBlock * nRowsInBlock;
        size_t nRows          = nRowsInBlock;
        if (startRow + nRows > nVectors) nRows = nVectors - startRow;

        TSubTask * local = lsTask.local();
        if (!local)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }
        DAAL_LS_RELEASE(TSubTask, lsTask, local); //releases local storage when leaving this scope

        safeStat |= local->getBlockOfRowsOfResults(pred, nFeatures, startRow, nRows, nClasses, nonEmptyClassMap, model, nIter, eps);
    });

    lsTask.reduce([=](SubTask<algorithmFPType, cpu> * local) { delete local; });
    return safeStat.detach();
}

/** Compute matrix Q from the 2-class parobabilities */
template <typename algorithmFPType, CpuType cpu>
inline void computeQ(size_t nClasses, const algorithmFPType * rProb, algorithmFPType * Q)
{
    algorithmFPType zero = 0.0;
    for (size_t i = 0; i < nClasses; i++)
    {
        Q[i * nClasses + i] = zero;
        for (size_t j = 0; j < i; j++)
        {
            algorithmFPType rProbJI = rProb[j * nClasses + i];
            Q[i * nClasses + i] += rProbJI * rProbJI;
            Q[i * nClasses + j] = -rProb[i * nClasses + j] * rProbJI;
            Q[j * nClasses + i] = Q[i * nClasses + j];
        }
        for (size_t j = i + 1; j < nClasses; j++)
        {
            algorithmFPType rProbJI = rProb[j * nClasses + i];
            Q[i * nClasses + i] += rProbJI * rProbJI;
        }
    }
}

/** Calculate objective function of the Algorithm 2 from [1] */
template <typename algorithmFPType, CpuType cpu>
inline algorithmFPType computeObjFunc(size_t nClasses, algorithmFPType * p, algorithmFPType * rProb)
{
    algorithmFPType objFunc = 0.0;
    for (size_t i = 0; i < nClasses; i++)
    {
        algorithmFPType diff;
        for (size_t j = 0; j < i; j++)
        {
            diff = rProb[i * nClasses + j] * p[i] + rProb[j * nClasses + i] * p[j];
            objFunc += diff * diff;
        }
        for (size_t j = i + 1; j < nClasses; j++)
        {
            diff = rProb[i * nClasses + j] * p[i] + rProb[j * nClasses + i] * p[j];
            objFunc += diff * diff;
        }
    }
    return objFunc;
}

template <typename algorithmFPType, CpuType cpu>
inline void updateProbabilities(size_t nClasses, const algorithmFPType * Q, algorithmFPType * Qp, algorithmFPType * p)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one  = 1.0;

    /* Calculate Q*p */
    for (size_t i = 0; i < nClasses; i++)
    {
        Qp[i] = zero;
        for (size_t j = 0; j < nClasses; j++)
        {
            Qp[i] += Q[i * nClasses + j] * p[j];
        }
    }

    /* Calculate p'*Q*p */
    algorithmFPType pQp = zero;
    for (size_t j = 0; j < nClasses; j++)
    {
        pQp += p[j] * Qp[j];
    }

    /* Update probabilities p */
    algorithmFPType sumP = zero;
    for (size_t j = 0; j < nClasses; j++)
    {
        p[j] = (pQp - Qp[j] + Q[j * nClasses + j] * p[j]) / Q[j * nClasses + j];
        sumP += p[j];
    }

    /* Normalize probabilities */
    algorithmFPType invSumP = one / sumP;
    for (size_t j = 0; j < nClasses; j++)
    {
        p[j] *= invSumP;
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubTask<algorithmFPType, cpu>::getBlockOfRowsOfResults(NumericTable * pred, size_t nFeatures, size_t startRow, size_t nRows,
                                                                        size_t nClasses, const size_t * nonEmptyClassMap, Model * model, size_t nIter,
                                                                        double eps)
{
    const algorithmFPType one(1.0);
    const algorithmFPType invNClasses(one / algorithmFPType(nClasses));

    algorithmFPType * rProb = _buffer.get();
    algorithmFPType * Q     = rProb + nRows * nClasses * nClasses;
    algorithmFPType * Qp    = Q + nClasses * nClasses;
    algorithmFPType * p     = Qp + nClasses;
    algorithmFPType * y     = p + nClasses;

    /* Get 2-class probabilities */
    services::Status s;
    DAAL_CHECK_STATUS(s, get2ClassProbabilities(nFeatures, startRow, nRows, nClasses, nonEmptyClassMap, y, model, rProb));

    algorithmFPType * rProbPtr = rProb;
    size_t nClassesSq          = nClasses * nClasses;
    for (size_t k = 0; k < nRows; k++, rProbPtr += nClassesSq)
    {
        /* Set initial probabilities */
        for (size_t j = 0; j < nClasses; j++)
        {
            p[j] = invNClasses;
        }

        /* Calculate matrix Q */
        computeQ<algorithmFPType, cpu>(nClasses, rProbPtr, Q);

        algorithmFPType objFuncPrev;
        objFuncPrev = daal::services::internal::MaxVal<algorithmFPType>::get();
        for (size_t it = 0; it < nIter; it++)
        {
            /* Check convergence criteria */
            algorithmFPType objFunc = computeObjFunc<algorithmFPType, cpu>(nClasses, p, rProbPtr);
            if (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(objFunc - objFuncPrev) < eps) break;
            objFuncPrev = objFunc;

            /* Update multiclass probabilities estimates */
            updateProbabilities<algorithmFPType, cpu>(nClasses, Q, Qp, p);
        }

        if (pred)
        {
            /* Calculate resulting classes labels */
            _mtR.set(pred, 0, startRow + k, 1);
            DAAL_CHECK_BLOCK_STATUS(_mtR);
            int & label             = *_mtR.get();
            algorithmFPType maxProb = p[0];
            DAAL_CHECK(nonEmptyClassMap[0] <= services::internal::MaxVal<int>::get(), services::ErrorIncorrectNumberOfClasses)
            label = (int)nonEmptyClassMap[0];
            for (size_t j = 1; j < nClasses; j++)
            {
                if (p[j] > maxProb)
                {
                    maxProb = p[j];
                    DAAL_ASSERT(nonEmptyClassMap[j] <= services::internal::MaxVal<int>::get())
                    label = (int)nonEmptyClassMap[j];
                }
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubTask<algorithmFPType, cpu>::get2ClassProbabilities(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                                                                       const size_t * nonEmptyClassMap, algorithmFPType * y, Model * model,
                                                                       algorithmFPType * rProb)
{
    NumericTablePtr xTable;
    services::Status s;
    DAAL_CHECK_STATUS(s, getInput(nFeatures, startRow, nRows, xTable));
    return predictSimpleClassifier(nFeatures, startRow, nRows, nClasses, nonEmptyClassMap, y, model, rProb, xTable);
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubTask<algorithmFPType, cpu>::predictSimpleClassifier(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                                                                        const size_t * nonEmptyClassMap, algorithmFPType * y, Model * model,
                                                                        algorithmFPType * rProb, const NumericTablePtr & xTable)
{
    services::Status status;
    NumericTablePtr yTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(y, 1, nRows, &status);
    DAAL_CHECK_STATUS_VAR(status);
    services::SharedPtr<typename classifier::prediction::Batch::ResultType> yRes(new typename classifier::prediction::Batch::ResultType());
    DAAL_CHECK_MALLOC(yTable.get() && yRes.get());
    yRes->set(classifier::prediction::prediction, yTable);
    const algorithmFPType one(1.0);
    for (size_t i = 1; i < nClasses; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            /* Compute prediction of the "simple" classifier for pair of labels (i, j) */
            const size_t imodel                   = ((nonEmptyClassMap[i] - 1) * nonEmptyClassMap[i]) / 2 + nonEmptyClassMap[j];
            classifier::prediction::Input * input = _simplePrediction->getInput();
            DAAL_CHECK(input, services::ErrorNullInput);
            input->set(classifier::prediction::data, xTable);
            input->set(classifier::prediction::model, model->getTwoClassClassifierModel(imodel));

            _simplePrediction->setResult(yRes);
            services::Status s = _simplePrediction->computeNoThrow();
            if (!s) return services::Status(services::ErrorMultiClassFailedToComputeTwoClassPrediction).add(s);

            /* Use sigmoid to calculate probabilities */
            daal::internal::MathInst<algorithmFPType, cpu>::vExp(nRows, y, y);
            algorithmFPType * rProbPtr = rProb;
            for (size_t k = 0; k < nRows; k++, rProbPtr += nClasses * nClasses)
            {
                algorithmFPType p          = one / (one + y[k]);
                rProbPtr[i * nClasses + j] = one - p;
                rProbPtr[j * nClasses + i] = p;
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubTaskCSR<algorithmFPType, cpu>::getInput(size_t nFeatures, size_t startRow, size_t nRows, NumericTablePtr & res)
{
    _mtX.next(startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_mtX);
    services::Status s;
    res = CSRNumericTable::create(const_cast<algorithmFPType *>(_mtX.values()), _mtX.cols(), _mtX.rows(), nFeatures, nRows,
                                  CSRNumericTableIface::CSRIndexing::oneBased, &s);
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubTaskDense<algorithmFPType, cpu>::getInput(size_t nFeatures, size_t startRow, size_t nRows, NumericTablePtr & res)
{
    _mtX.next(startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_mtX);
    services::Status st;
    res = HomogenNumericTableCPU<algorithmFPType, cpu>::create(const_cast<algorithmFPType *>(_mtX.get()), nFeatures, nRows, &st);
    return st;
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
