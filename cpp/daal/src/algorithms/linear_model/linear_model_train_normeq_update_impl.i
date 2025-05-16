/* file: linear_model_train_normeq_update_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "src/algorithms/linear_model/linear_model_train_normeq_kernel.h"
#include "src/algorithms/linear_model/linear_model_hyperparameter_impl.h"

#include "src/externals/service_blas.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"
#include "src/services/service_profiler.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
ThreadingTask<algorithmFPType, cpu>::ThreadingTask(size_t nBetasIntercept, size_t nResponses, Status & st)
    : _nBetasIntercept(nBetasIntercept), _nResponses(nResponses)
{
    _xtx = service_scalable_calloc<algorithmFPType, cpu>(nBetasIntercept * nBetasIntercept);
    _xty = service_scalable_calloc<algorithmFPType, cpu>(nBetasIntercept * nResponses);
    if (!_xtx || !_xty) st.add(ErrorMemoryAllocationFailed);
}

template <typename algorithmFPType, CpuType cpu>
ThreadingTask<algorithmFPType, cpu> * ThreadingTask<algorithmFPType, cpu>::create(size_t nBetasIntercept, size_t nResponses)
{
    Status st;
    ThreadingTask<algorithmFPType, cpu> * res = new ThreadingTask<algorithmFPType, cpu>(nBetasIntercept, nResponses, st);
    if (!st)
    {
        delete res;
        return nullptr;
    }
    return res;
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::update(DAAL_INT startRow, DAAL_INT nRows, const NumericTable & xTable, const NumericTable & yTable)
{
    DAAL_INT nFeatures(xTable.getNumberOfColumns());

    /* SYRK and GEMM parameters */
    char up      = 'U';
    char trans   = 'T';
    char notrans = 'N';
    algorithmFPType alpha(1.0);

    _xBlock.set(const_cast<NumericTable &>(xTable), startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_xBlock);
    const algorithmFPType * x = _xBlock.get();

    _yBlock.set(const_cast<NumericTable &>(yTable), startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_yBlock);
    const algorithmFPType * y = _yBlock.get();

    {
        DAAL_PROFILER_THREADING_TASK(update.syrkX);
        BlasInst<algorithmFPType, cpu>::xxsyrk(&up, &notrans, &nFeatures, &nRows, &alpha, const_cast<algorithmFPType *>(x), &nFeatures, &alpha, _xtx,
                                               &_nBetasIntercept);
    }

    if (nFeatures < _nBetasIntercept)
    {
        DAAL_PROFILER_THREADING_TASK(update.gemm1X);
        algorithmFPType * xtxPtr     = _xtx + nFeatures * _nBetasIntercept;
        const algorithmFPType * xPtr = x;

        for (DAAL_INT i = 0; i < nRows; i++, xPtr += nFeatures)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (DAAL_INT j = 0; j < nFeatures; j++)
            {
                xtxPtr[j] += xPtr[j];
            }
        }

        xtxPtr[nFeatures] += algorithmFPType(nRows);
    }

    {
        DAAL_PROFILER_THREADING_TASK(update.gemmXY);
        BlasInst<algorithmFPType, cpu>::xxgemm(&notrans, &trans, &nFeatures, &_nResponses, &nRows, &alpha, x, &nFeatures, y, &_nResponses, &alpha,
                                               _xty, &_nBetasIntercept);
    }

    if (nFeatures < _nBetasIntercept)
    {
        DAAL_PROFILER_THREADING_TASK(update.gemm1Y);
        const algorithmFPType * yPtr = y;
        for (DAAL_INT i = 0; i < nRows; i++, yPtr += _nResponses)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (DAAL_INT j = 0; j < _nResponses; j++)
            {
                _xty[j * _nBetasIntercept + nFeatures] += yPtr[j];
            }
        }
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
void ThreadingTask<algorithmFPType, cpu>::reduce(algorithmFPType * xtx, algorithmFPType * xty)
{
    {
        DAAL_PROFILER_THREADING_TASK(reduce.syrkX);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < (_nBetasIntercept * _nBetasIntercept); i++)
        {
            xtx[i] += _xtx[i];
        }
    }

    {
        DAAL_PROFILER_THREADING_TASK(reduce.gemmXY);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < (_nBetasIntercept * _nResponses); i++)
        {
            xty[i] += _xty[i];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
ThreadingTask<algorithmFPType, cpu>::~ThreadingTask()
{
    service_scalable_free<algorithmFPType, cpu>(_xtx);
    service_scalable_free<algorithmFPType, cpu>(_xty);
}

template <typename algorithmFPType, CpuType cpu>
Status UpdateKernel<algorithmFPType, cpu>::compute(const NumericTable & xTable, const NumericTable & yTable, NumericTable & xtxTable,
                                                   NumericTable & xtyTable, bool initializeResult, bool interceptFlag)
{
    return UpdateKernel<algorithmFPType, cpu>::compute(xTable, yTable, xtxTable, xtyTable, initializeResult, interceptFlag, nullptr);
}

template <typename algorithmFPType, CpuType cpu>
Status computeNonBatchedAggregates(const DAAL_INT nRows, const DAAL_INT nCols, const DAAL_INT nResponses, bool initializeResult, bool interceptFlag,
                                   const algorithmFPType * xPtr, const algorithmFPType * yPtr, algorithmFPType * xtx, algorithmFPType * xty)
{
    DAAL_INT nBetasIntercept            = nCols + static_cast<int>(interceptFlag);
    DAAL_INT one_int                    = 1;
    algorithmFPType one                 = 1;
    algorithmFPType zero                = 0;
    algorithmFPType * mult_current_data = initializeResult ? &zero : &one;
    daal::services::internal::TArray<algorithmFPType, cpu> ones;
    if (interceptFlag)
    {
        ones = daal::services::internal::TArray<algorithmFPType, cpu>(nRows);
        std::fill(ones.get(), ones.get() + nRows, algorithmFPType(1));
    }

    BlasInst<algorithmFPType, cpu>::xsyrk("U", "N", &nCols, &nRows, &one, xPtr, &nCols, mult_current_data, xtx, &nBetasIntercept);
    if (interceptFlag)
    {
        BlasInst<algorithmFPType, cpu>::xgemv("N", &nCols, &nRows, &one, xPtr, &nCols, ones.get(), &one_int, mult_current_data,
                                              xtx + static_cast<size_t>(nBetasIntercept) * static_cast<size_t>(nCols), &one_int);
        const size_t idx_last = static_cast<size_t>(nBetasIntercept) * static_cast<size_t>(nBetasIntercept) - 1;
        if (initializeResult)
            xtx[idx_last] = nRows;
        else
            xtx[idx_last] += nRows;
    }

    if (nResponses == 1)
    {
        BlasInst<algorithmFPType, cpu>::xgemv("N", &nCols, &nRows, &one, xPtr, &nCols, yPtr, &one_int, mult_current_data, xty, &one_int);
        if (interceptFlag)
        {
            const algorithmFPType last_val = BlasInst<algorithmFPType, cpu>::xxdot(&nRows, yPtr, &one_int, ones.get(), &one_int);
            if (initializeResult)
                xty[nCols] = last_val;
            else
                xty[nCols] += last_val;
        }
    }
    else
    {
        BlasInst<algorithmFPType, cpu>::xgemm("N", "T", &nCols, &nResponses, &nRows, &one, xPtr, &nCols, yPtr, &nResponses, mult_current_data, xty,
                                              &nBetasIntercept);
        if (interceptFlag)
        {
            BlasInst<algorithmFPType, cpu>::xgemv("N", &nResponses, &nRows, &one, yPtr, &nResponses, ones.get(), &one_int, mult_current_data,
                                                  xty + nCols, &nBetasIntercept);
        }
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status UpdateKernel<algorithmFPType, cpu>::compute(const NumericTable & xTable, const NumericTable & yTable, NumericTable & xtxTable,
                                                   NumericTable & xtyTable, bool initializeResult, bool interceptFlag,
                                                   const HyperparameterType * hyperparameter)
{
    DAAL_PROFILER_TASK(computeUpdate);
    DAAL_INT nRows(xTable.getNumberOfRows());         /* observations */
    DAAL_INT nResponses(yTable.getNumberOfColumns()); /* responses */
    DAAL_INT nBetas(xTable.getNumberOfColumns() + 1); /* coefficients */

    size_t nBetasIntercept = (interceptFlag ? nBetas : (nBetas - 1));

    WriteRowsType xtxBlock(xtxTable, 0, nBetasIntercept);
    DAAL_CHECK_BLOCK_STATUS(xtxBlock);
    algorithmFPType * xtx = xtxBlock.get();

    WriteRowsType xtyBlock(xtyTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(xtyBlock);
    algorithmFPType * xty = xtyBlock.get();

    /// Logic here is as follows: it needs to compute t(X)*X and t(X)*y.
    /// If both are done together, it's possible to reuse caches of data to speed up computations,
    /// which the code here does by dividing the data into batches of rows on which both aggregates
    /// are computed, with the batches processed in parallel. But as the number of columns in the
    /// data grows, the potential speed gains from calculating both aggregates simultaneously
    /// decreases, and the memory requirements increase, which can become a problem when there are
    /// many threads in the system. Hence, if the number of columns is too large, it will compute
    /// both aggregates independently, in separate calls to BLAS functions, while if the number of
    /// columns is reasonably small, will prefer the batched procedure which typically ends up
    /// being faster.
    DAAL_INT64 maxColsBatched          = 4096;
    DAAL_INT64 smallRowsThreshold      = 10000;
    DAAL_INT64 smallRowsMaxColsBatched = 1024;
    if (hyperparameter != nullptr)
    {
        services::Status status = hyperparameter->find(denseUpdateMaxColsBatched, maxColsBatched);
        DAAL_CHECK(maxColsBatched > 0, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);

        status = hyperparameter->find(denseSmallRowsThreshold, smallRowsThreshold);
        DAAL_CHECK(smallRowsThreshold > 0, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);

        status = hyperparameter->find(denseSmallRowsMaxColsBatched, smallRowsMaxColsBatched);
        DAAL_CHECK(smallRowsMaxColsBatched > 0, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);
    }

    const bool use_non_batched_route = nBetas >= maxColsBatched || (nRows >= smallRowsThreshold && nBetas >= smallRowsMaxColsBatched);
    if (use_non_batched_route)
    {
        const DAAL_INT nCols = xTable.getNumberOfColumns();
        ReadRowsType xBlock(const_cast<NumericTable &>(xTable), 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(xBlock);
        ReadRowsType yBlock(const_cast<NumericTable &>(yTable), 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(yBlock);
        const algorithmFPType * xPtr = xBlock.get();
        const algorithmFPType * yPtr = yBlock.get();
        return computeNonBatchedAggregates<algorithmFPType, cpu>(nRows, nCols, nResponses, initializeResult, interceptFlag, xPtr, yPtr, xtx, xty);
    }

    /* Initialize output arrays by zero in case of batch mode */
    if (initializeResult)
    {
        service_memset<algorithmFPType, cpu>(xtx, 0, nBetasIntercept * nBetasIntercept);
        service_memset<algorithmFPType, cpu>(xty, 0, nResponses * nBetasIntercept);
    }

    /* Split rows by blocks */
    size_t nRowsInBlock = 128;
    if (hyperparameter != nullptr)
    {
        DAAL_INT64 nRowsInBlockInt64 = 0l;
        services::Status status      = hyperparameter->find(denseUpdateStepBlockSize, nRowsInBlockInt64);
        DAAL_CHECK(0l < nRowsInBlockInt64, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);

        nRowsInBlock = static_cast<size_t>(nRowsInBlockInt64);
    }

    size_t nBlocks = nRows / nRowsInBlock;
    nBlocks += bool(nRows % nRowsInBlock);

    /* Create TLS */
    daal::static_tls<ThreadingTaskType *> tls([=]() -> ThreadingTaskType * { return ThreadingTaskType::create(nBetasIntercept, nResponses); });

    SafeStatus safeStat;
    daal::static_threader_for(nBlocks, [=, &tls, &xTable, &yTable, &safeStat](int iBlock, size_t tid) {
        ThreadingTaskType * tlsLocal = tls.local(tid);

        if (!tlsLocal)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }

        size_t startRow = iBlock * nRowsInBlock;
        size_t endRow   = startRow + nRowsInBlock;
        if (endRow > nRows)
        {
            endRow = nRows;
        }

        Status localSt = tlsLocal->update(startRow, endRow - startRow, xTable, yTable);
        DAAL_CHECK_STATUS_THR(localSt);
    });

    Status st = safeStat.detach();
    tls.reduce([=, &st](ThreadingTaskType * tlsLocal) -> void {
        if (!tlsLocal) return;
        if (st) tlsLocal->reduce(xtx, xty);
        delete tlsLocal;
    });

    return st;
}

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
