/* file: logitboost_train_friedman_impl.i */
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
//  Common functions for Logit Boost model training
//--
*/
/*
//
//  REFERENCES
//
//  1. J. Friedman, T. Hastie, R. Tibshirani.
//     Additive logistic regression: a statistical view of boosting,
//     The annals of Statistics, 2000, v28 N2, pp. 337-407
//  2. J. Friedman, T. Hastie, R. Tibshirani.
//     The Elements of Statistical Learning:
//     Data Mining, Inference, and Prediction,
//     Springer, 2001.
//
*/

#ifndef __LOGITBOOST_TRAIN_FRIEDMAN_IMPL_I__
#define __LOGITBOOST_TRAIN_FRIEDMAN_IMPL_I__

#include "src/threading/threading.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/algorithms/service_threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/logitboost/logitboost_impl.i"
#include "src/algorithms/logitboost/logitboost_train_friedman_aux.i"

using namespace daal::algorithms::logitboost::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class LogitBoostLs
{
private:
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    typedef typename daal::services::SharedPtr<daal::algorithms::regression::training::Batch> TrainLernerPtr;
    typedef typename daal::services::SharedPtr<daal::algorithms::regression::prediction::Batch> PredictLernerPtr;

public:
    LogitBoostLs(const size_t n) : _nRows(n), _isInit(false) {}
    ~LogitBoostLs() {}
    DAAL_NEW_DELETE();

    services::Status allocate(NumericTablePtr & x, TrainLernerPtr & train, PredictLernerPtr & predict)
    {
        services::Status status;
        if (!_isInit)
        {
            _learnerTrain   = train->clone();
            _learnerPredict = predict->clone();
            if (!wArray) wArray = HomogenNT::create(1, _nRows, &status);
            if (!zArray) zArray = HomogenNT::create(1, _nRows, &status);

            auto rpr = new regression::prediction::Result();
            DAAL_CHECK_MALLOC(rpr)
            _predictionRes.reset(rpr);

            regression::training::Input * input       = _learnerTrain->getInput();
            regression::prediction::Input * predInput = _learnerPredict->getInput();
            if (!input || !predInput)
                status.add(services::ErrorNullInput);
            else
            {
                input->set(regression::training::dependentVariables, zArray);
                input->set(regression::training::weights, wArray);
                input->set(regression::training::data, x);
                predInput->set(regression::prediction::data, x);
            }
        }

        _isInit = true;
        return status;
    }

    services::Status allocate()
    {
        services::Status status;
        if (!wArray) wArray = HomogenNT::create(1, _nRows, &status);
        if (!zArray) zArray = HomogenNT::create(1, _nRows, &status);
        return status;
    }

    services::Status run(const size_t classIdx, data_management::DataCollection & models, TArray<algorithmFPType, cpu> & pred)
    {
        _learnerTrain->resetResult();
        services::Status status = _learnerTrain->computeNoThrow();
        DAAL_CHECK_STATUS_VAR(status);

        regression::ModelPtr learnerModel = _learnerTrain->getResult()->get(regression::training::model);
        models[classIdx]                  = learnerModel;

        regression::prediction::Input * predInput = _learnerPredict->getInput();
        DAAL_CHECK(predInput, services::ErrorNullInput);
        predInput->set(regression::prediction::model, learnerModel);

        HomogenNTPtr predTable(HomogenNT::create(pred.get() + classIdx * _nRows, 1, _nRows, &status));
        DAAL_CHECK_STATUS_VAR(status);

        _predictionRes->set(regression::prediction::prediction, predTable);
        status = _learnerPredict->setResult(_predictionRes);
        DAAL_CHECK_STATUS_VAR(status);

        status = _learnerPredict->computeNoThrow();
        return status;
    }

public:
    HomogenNTPtr wArray;
    HomogenNTPtr zArray;

private:
    TrainLernerPtr _learnerTrain;
    PredictLernerPtr _learnerPredict;
    regression::prediction::ResultPtr _predictionRes;
    const size_t _nRows;
    bool _isInit;
};

template <typename algorithmFPType, CpuType cpu>
services::Status UpdateFPNew(size_t nc, size_t n, algorithmFPType * F, algorithmFPType * P, const algorithmFPType * pred,
                             daal::ls<LogitBoostLs<algorithmFPType, cpu> *> & lsData)
{
    const size_t minBlockSize = 768;
    const size_t nThreads     = threader_get_threads_number();
    size_t nBlocks            = n / minBlockSize;
    nBlocks                   = nBlocks ? nBlocks : 1;
    nBlocks                   = nBlocks < nThreads ? nBlocks : nThreads;
    const size_t blockSize    = n / nBlocks;
    const size_t tail         = n - nBlocks * blockSize;

    const algorithmFPType inv_nc = 1.0 / (algorithmFPType)nc;
    const algorithmFPType coef   = (algorithmFPType)(nc - 1) / (algorithmFPType)nc;

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * blockSize + (iBlock < tail ? iBlock : tail);
        const size_t size  = (iBlock < tail) ? blockSize + 1 : blockSize;

        /* Update additive function's values
           Step 2.b) of the Algorithm 6 from [1] */
        /* i-row contains Fi() for all classes in i-th point x */
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = start; i < start + size; i++)
        {
            algorithmFPType s = 0.0;
            for (size_t k = 0; k < nc; k++)
            {
                s += pred[k * n + i];
            }

            for (size_t j = 0; j < nc; j++)
            {
                F[i * nc + j] += coef * (pred[j * n + i] - s * inv_nc);
            }
        }

        struct LogitBoostLs<algorithmFPType, cpu> * lsLocal = lsData.local();
        if (!lsLocal)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }
        DAAL_CHECK_STATUS_THR(lsLocal->allocate());
        algorithmFPType * buffer = lsLocal->wArray->getArray();
        if (!buffer)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }

        /* Update probabilities
           Step 2.c) of the Algorithm 6 from [1] */
        const bool useFullBuffer = size * nc <= n;
        if (useFullBuffer) daal::internal::MathInst<algorithmFPType, cpu>::vExp(nc * size, F + start * nc, buffer);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < size; i++)
        {
            const size_t offset = useFullBuffer ? i * nc : 0;
            if (!useFullBuffer) daal::internal::MathInst<algorithmFPType, cpu>::vExp(nc, F + (i + start) * nc, buffer);

            algorithmFPType s = 0.0;

            for (size_t j = 0; j < nc; j++)
            {
                s += buffer[offset + j];
            }
            // if low accuracy exp() returns NaN\Inf - convert it to some positive big value
            s = services::internal::infToBigValue<cpu>(s);

            algorithmFPType invs = (algorithmFPType)1.0 / s;

            const size_t row = i + start;
            for (size_t j = 0; j < nc; j++)
            {
                // Normalize probabilities
                P[j * n + row] = buffer[offset + j] * invs;
            }
        }
    });

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
services::Status LogitBoostTrainKernel<friedman, algorithmFPType, cpu>::compute(const size_t na, NumericTablePtr a[], Model * r,
                                                                                const Parameter * par)
{
    const algorithmFPType zero(0.0);
    const algorithmFPType fp_one(1.0);
    Parameter * parameter = const_cast<Parameter *>(par);
    NumericTablePtr x     = a[0];
    NumericTablePtr y     = a[1];
    r->setNFeatures(x->getNumberOfColumns());

    algorithmFPType acc        = parameter->accuracyThreshold;
    const size_t M             = parameter->maxIterations;
    const size_t nc            = parameter->nClasses;
    const algorithmFPType thrW = (algorithmFPType)(parameter->weightsDegenerateCasesThreshold);
    const algorithmFPType thrZ = (algorithmFPType)(parameter->responsesDegenerateCasesThreshold);
    const size_t n             = x->getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nc);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nc, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> pred(n * nc);
    TArray<algorithmFPType, cpu> F(n * nc);
    TArray<algorithmFPType, cpu> P(n * nc);

    DAAL_CHECK(pred.get() && F.get() && P.get(), services::ErrorMemoryAllocationFailed);

    services::Status s;
    HomogenNTPtr wTable(HomogenNT::create(1, n, &s));
    DAAL_CHECK_STATUS_VAR(s);
    HomogenNTPtr zTable(HomogenNT::create(1, n, &s));
    DAAL_CHECK_STATUS_VAR(s);

    algorithmFPType * w = wTable->getArray();

    const algorithmFPType inv_n  = fp_one / (algorithmFPType)n;
    const algorithmFPType inv_nc = fp_one / (algorithmFPType)nc;

    /* Initialize weights, probs and additive function values.
       Step 1) of the Algorithm 6 from [1] */
    for (size_t i = 0; i < n; i++)
    {
        w[i] = inv_n;
    }
    for (size_t i = 0; i < n * nc; i++)
    {
        P[i] = inv_nc;
    }
    algorithmFPType logL   = -algorithmFPType(n) * daal::internal::MathInst<algorithmFPType, cpu>::sLog(inv_nc);
    algorithmFPType accCur = daal::services::internal::MaxVal<algorithmFPType>::get();

    for (size_t i = 0; i < n * nc; i++)
    {
        F[i] = zero;
    }

    ReadColumns<int, cpu> yCols(*y, 0, 0, n);
    DAAL_CHECK_BLOCK_STATUS(yCols);
    const int * y_label = yCols.get();
    DAAL_ASSERT(y_label);

    services::SharedPtr<regression::training::Batch> learnerTrain     = parameter->weakLearnerTraining;
    services::SharedPtr<regression::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;

    /* Clear the collection of weak learners models in the boosting model */
    r->clearWeakLearnerModels();
    data_management::DataCollection models(nc);

    SafeStatus safeStat;
    daal::ls<LogitBoostLs<algorithmFPType, cpu> *> lsData([&]() {
        auto ptr = new LogitBoostLs<algorithmFPType, cpu>(n);
        if (!ptr)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return ptr;
    });

    /* Repeat for m = 0, 1, ..., M-1
       Step 2) of the Algorithm 6 from [1] */
    for (size_t m = 0; m < M; m++)
    {
        /* Repeat for j = 0, 1, ..., nk-1
           Step 2.a) of the Algorithm 6 from [1] */
        daal::threader_for(nc, nc, [&](size_t j) {
            struct LogitBoostLs<algorithmFPType, cpu> * lsLocal = lsData.local();
            if (!lsLocal) return;

            services::Status localStatus = lsLocal->allocate(x, learnerTrain, learnerPredict);
            DAAL_CHECK_STATUS_THR(localStatus);

            initWZ<algorithmFPType, cpu>(n, nc, j, y_label, P.get(), thrW, lsLocal->wArray->getArray(), thrZ, lsLocal->zArray->getArray());

            localStatus = lsLocal->run(j, models, pred);
            DAAL_CHECK_STATUS_THR(localStatus);
        });
        DAAL_CHECK_SAFE_STATUS();

        for (size_t j = 0; j < nc; ++j)
        {
            r->addWeakLearnerModel(services::staticPointerCast<regression::Model, SerializationIface>(models[j]));
        }

        /* Update additive function's values and probabilities
           Step 2.b and 2.c) of the Algorithm 6 from [1] */
        s |= UpdateFPNew<algorithmFPType, cpu>(nc, n, F.get(), P.get(), pred.get(), lsData);
        DAAL_CHECK_STATUS_VAR(s);
        /* Calculate model accuracy */
        calculateAccuracy<algorithmFPType, cpu>(n, nc, y_label, P.get(), logL, accCur);

        if (accCur < acc)
        {
            r->setIterations(m + 1);
            break;
        }
    }
    s |= safeStat.detach();

    lsData.reduce([=](LogitBoostLs<algorithmFPType, cpu> * logitBoostLs) { delete logitBoostLs; });
    return s;
}

} // namespace internal
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
