/* file: logistic_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for logistic regression classification predictions calculation
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/logistic_regression/logistic_regression_predict_kernel.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/algorithms/logistic_regression/logistic_regression_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_algo_utils.h"
#include "src/services/service_environment.h"
#include "src/externals/service_blas.h"
#include "src/algorithms/objective_function/cross_entropy_loss/cross_entropy_loss_dense_default_batch_kernel.h"
#include "src/algorithms/objective_function/logistic_loss/logistic_loss_dense_default_batch_kernel.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{
namespace ll = daal::algorithms::optimization_solver::logistic_loss;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictBinaryClassificationTask
{
public:
    PredictBinaryClassificationTask(const NumericTable * x, NumericTable * y, NumericTable * prob, NumericTable * logProb)
        : _data(x), _res(y), _prob(prob), _logProb(logProb)
    {}
    services::Status run(const NumericTable & beta, services::HostAppIface * pHostApp)
    {
        const size_t nCols        = _data->getNumberOfColumns();
        const size_t nRowsTotal   = _data->getNumberOfRows();
        const size_t nRowsInBlock = 512;
        const size_t nDataBlocks  = nRowsTotal / nRowsInBlock + !!(nRowsTotal % nRowsInBlock);

        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nCols, size_t(1));
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsTotal, size_t(2));
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsTotal, nCols + 1);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsInBlock, sizeof(algorithmFPType));

        ReadRows<algorithmFPType, cpu> betaBD(const_cast<NumericTable &>(beta), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(betaBD);

        SafeStatus safeStat;
        HostAppHelper host(pHostApp, 1000);

        StaticTlsMem<algorithmFPType, cpu> bufferTls(nRowsInBlock);

        daal::static_threader_for(nDataBlocks, [&](size_t iBlock, size_t tid) {
            services::Status s;
            if (host.isCancelled(s, 1))
            {
                safeStat.add(s);
                return;
            }
            const size_t iStartRow      = iBlock * nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? nRowsTotal - iBlock * nRowsInBlock : nRowsInBlock;
            algorithmFPType * buff      = bufferTls.local(tid);
            DAAL_CHECK_MALLOC_THR(buff);

            DAAL_CHECK_STATUS_THR(applyBetaImpl(_data, betaBD.get(), buff, nRowsToProcess, nCols, iStartRow, true));

            if (_res)
            {
                WriteOnlyRows<algorithmFPType, cpu> resBD(_res, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(resBD);
                predictLabels(buff, resBD.get(), nRowsToProcess);
            }

            if (_prob || _logProb)
            {
                ll::internal::LogLossKernel<algorithmFPType, ll::defaultDense, cpu>::sigmoid(buff, buff, nRowsToProcess);
            }

            if (_prob || _logProb)
            {
                auto ntForProb = _prob ? _prob : _logProb;
                WriteOnlyRows<algorithmFPType, cpu> probBD(ntForProb, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(probBD);
                algorithmFPType * prob = probBD.get();

                if (ntForProb->getNumberOfColumns() == 2)
                {
                    for (size_t i = 0; i < nRowsToProcess; ++i)
                    {
                        prob[i * 2 + 0] = algorithmFPType(1.0) - buff[i];
                        prob[i * 2 + 1] = buff[i];
                    }
                }
                else // for backward compatibility with old interfaces
                {
                    services::internal::tmemcpy<algorithmFPType, cpu>(prob, buff, nRowsToProcess);
                }

                if (_prob && _logProb)
                {
                    WriteOnlyRows<algorithmFPType, cpu> logBD(_logProb, iStartRow, nRowsToProcess);
                    DAAL_CHECK_BLOCK_STATUS_THR(logBD);
                    daal::internal::MathInst<algorithmFPType, cpu>::vLog(nRowsToProcess * _logProb->getNumberOfColumns(), prob, logBD.get());
                }
                else if (!_prob && _logProb)
                {
                    daal::internal::MathInst<algorithmFPType, cpu>::vLog(nRowsToProcess * _logProb->getNumberOfColumns(), prob, prob);
                }
            }
        });

        return safeStat.detach();
    }

protected:
    services::Status gemvSoa(const NumericTable * x, const algorithmFPType * b, algorithmFPType * res, size_t nRows, size_t nCols, size_t xOffset)
    {
        const DAAL_INT incX(1);
        const DAAL_INT incY(1);
        const DAAL_INT size = static_cast<DAAL_INT>(nRows);
        DAAL_ASSERT(nRows <= services::internal::MaxVal<DAAL_INT>::get());

        services::internal::service_memset_seq<algorithmFPType, cpu>(res, algorithmFPType(0.0), nRows);

        for (size_t i = 0; i < nCols; ++i)
        {
            ReadColumns<algorithmFPType, cpu> xBlock(const_cast<NumericTable *>(x), i, xOffset, nRows);
            DAAL_CHECK_BLOCK_STATUS(xBlock);
            const algorithmFPType * const xData = xBlock.get();
            const algorithmFPType value         = b[i];

            BlasInst<algorithmFPType, cpu>::xxaxpy(&size, &value, xData, &incX, res, &incY);
        }
        return services::Status();
    }

    services::Status applyBetaImpl(const NumericTable * x, const algorithmFPType * beta, algorithmFPType * xb, size_t nRows, size_t nCols,
                                   size_t xOffset, bool bIntercept)
    {
        services::Status s;

        if (dynamic_cast<SOANumericTable *>(const_cast<NumericTable *>(_data)))
        {
            s |= gemvSoa(x, beta + 1, xb, nRows, nCols, xOffset);
            if (bIntercept)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nRows; ++i) xb[i] += beta[0];
            }
        }
        else
        {
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), xOffset, nRows);
            DAAL_CHECK_BLOCK_STATUS(xBD);
            ll::internal::LogLossKernel<algorithmFPType, ll::defaultDense, cpu>::applyBeta(xBD.get(), beta, xb, nRows, nCols, bIntercept);
        }

        return s;
    }

    void predictLabels(const algorithmFPType * pRaw, algorithmFPType * pRes, size_t nRows)
    {
        const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
        //pRaw contains raw values of sigmoid argument
        for (size_t iRow = 0; iRow < nRows; ++iRow)
        {
            //probability is a sigmoid(f) hence sign(f) can be checked
            pRes[iRow] = label[services::internal::SignBit<algorithmFPType, cpu>::get(pRaw[iRow])];
        }
    }
    void stretchProbaOnTwoColumns(algorithmFPType * const data, const size_t numberRows)
    {
        for (size_t index = 2 * numberRows - 1; index >= 3; index -= 2)
        {
            data[index]     = data[index / 2];
            data[index - 1] = algorithmFPType(1.) - data[index];
        }
        data[1] = data[0];
        data[0] = algorithmFPType(1.) - data[1];
    }

protected:
    const NumericTable * _data;
    NumericTable * _res;
    NumericTable * _prob;
    NumericTable * _logProb;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictMulticlassTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictMulticlassTask
{
public:
    PredictMulticlassTask(const NumericTable * x, NumericTable * y, NumericTable * prob, NumericTable * logProb)
        : _data(x), _res(y), _prob(prob), _logProb(logProb)
    {}
    services::Status run(const NumericTable & beta, services::HostAppIface * pHostApp);

protected:
    void predictRaw(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * rawRes, size_t nRows, size_t nClasses, size_t nCols);

protected:
    const NumericTable * _data;
    NumericTable * _res;
    NumericTable * _prob;
    NumericTable * _logProb;
};

template <typename algorithmFPType, CpuType cpu>
struct TlsData
{
    DAAL_NEW_DELETE();
    TlsData(size_t n, const NumericTable * ntX) : x(const_cast<NumericTable *>(ntX))
    {
        raw = services::internal::service_scalable_calloc<algorithmFPType, cpu>(n);
    }

    ~TlsData()
    {
        if (raw) services::internal::service_scalable_free<algorithmFPType, cpu>(raw);
    }

    ReadRows<algorithmFPType, cpu> x;
    WriteOnlyRows<algorithmFPType, cpu> tmp;
    algorithmFPType * raw = nullptr;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::run(const NumericTable & beta, services::HostAppIface * pHostApp)
{
    const size_t nRowsTotal = _data->getNumberOfRows();
    const size_t nCols      = _data->getNumberOfColumns();
    const size_t nClasses   = beta.getNumberOfRows();
    DAAL_ASSERT(beta.getNumberOfColumns() == nCols + 1);
    const size_t nYPerRow            = nClasses;
    const size_t nRowsInBlockDefault = 500;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nCols, size_t(1));
    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nYPerRow, nCols);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nCols + nYPerRow, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsTotal, nClasses);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsTotal, nCols + 1);

    WriteOnlyRows<algorithmFPType, cpu> resBD;
    if (_res)
    {
        resBD.set(_res, 0, nRowsTotal);
        DAAL_CHECK_BLOCK_STATUS(resBD);
    }
    const size_t nRowsInBlock = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize() * 0.8,
                                                                              (nCols + nYPerRow) * sizeof(algorithmFPType), nRowsInBlockDefault);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsInBlock * nClasses, sizeof(algorithmFPType));

    const size_t nDataBlocks = nRowsTotal / nRowsInBlock + !!(nRowsTotal % nRowsInBlock);

    ReadRows<algorithmFPType, cpu> betaBD(const_cast<NumericTable &>(beta), 0, nClasses);
    DAAL_CHECK_BLOCK_STATUS(betaBD);

    using TlsDataCpu = TlsData<algorithmFPType, cpu>;
    daal::tls<TlsDataCpu *> tlsData([=]() -> TlsDataCpu * { return new TlsDataCpu(nRowsInBlock * nClasses, _data); });

    SafeStatus safeStat;
    HostAppHelper host(pHostApp, 1000);

    daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock) {
        services::Status s;
        if (host.isCancelled(s, 1))
        {
            safeStat.add(s);
            return;
        }
        const size_t iStartRow      = iBlock * nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? nRowsTotal - iBlock * nRowsInBlock : nRowsInBlock;

        TlsDataCpu * pLocal = tlsData.local();
        DAAL_CHECK_MALLOC_THR(pLocal);
        algorithmFPType * pRawValues = pLocal->raw;

        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        predictRaw(xBD.get(), betaBD.get(), pRawValues, nRowsToProcess, nClasses, nCols);

        if (_res)
        {
            algorithmFPType * res = resBD.get() + iStartRow;
            for (size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                res[iRow] = algorithmFPType(services::internal::getMaxElementIndex<algorithmFPType, cpu>(pRawValues + iRow * nClasses, nClasses));
        }
        namespace cel = daal::algorithms::optimization_solver::cross_entropy_loss;
        //softmax
        if (_prob && !_logProb)
        {
            //compute softmax and write results directly to _prob
            pLocal->tmp.set(_prob, iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
            cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::softmax(pRawValues, pLocal->tmp.get(), nRowsToProcess,
                                                                                                    nClasses, nullptr, nullptr);
        }
        else if (_prob || _logProb)
        {
            //compute softmax in pRawValues
            cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::softmax(pRawValues, pRawValues, nRowsToProcess, nClasses,
                                                                                                    nullptr, nullptr);
            if (_prob)
            {
                pLocal->tmp.set(_prob, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
                services::internal::tmemcpy<algorithmFPType, cpu>(pLocal->tmp.get(), pRawValues, nRowsToProcess * nClasses);
            }
            if (_logProb)
            {
                pLocal->tmp.set(_logProb, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
                daal::internal::MathInst<algorithmFPType, cpu>::vLog(nRowsToProcess * nClasses, pRawValues, pLocal->tmp.get());
            }
        }
    });
    tlsData.reduce([](TlsDataCpu * ptr) { delete ptr; });

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::predictRaw(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * rawRes,
                                                             size_t nRows, size_t nClasses, size_t nCols)
{
    namespace cel = daal::algorithms::optimization_solver::cross_entropy_loss;
    cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::applyBeta(x, beta, rawRes, nRows, nClasses, nCols, true);
}

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const logistic_regression::Model * m, size_t nClasses, NumericTable * pRes,
                                                                      NumericTable * pProbab, NumericTable * pLogProbab)
{
    const daal::algorithms::logistic_regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::logistic_regression::internal::ModelImpl *>(m);
    if (nClasses == 2)
    {
        PredictBinaryClassificationTask<algorithmFPType, cpu> task(x, pRes, pProbab, pLogProbab);
        return task.run(*pModel->getBeta(), pHostApp);
    }
    PredictMulticlassTask<algorithmFPType, cpu> task(x, pRes, pProbab, pLogProbab);
    return task.run(*pModel->getBeta(), pHostApp);
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace logistic_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif
