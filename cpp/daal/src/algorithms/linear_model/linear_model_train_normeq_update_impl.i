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
#include "services/internal/service_profiler.h"

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
LinearModelReducer<algorithmFPType, cpu>::LinearModelReducer(const NumericTable & xTable, const NumericTable & yTable, DAAL_INT nBetasIntercept,
                                                             DAAL_INT numRowsInBlock, size_t numBlocks)
    : _xTable(xTable),
      _yTable(yTable),
      _nBetasIntercept(nBetasIntercept),
      _numRowsInBlock(numRowsInBlock),
      _numBlocks(numBlocks),
      _nFeatures(xTable.getNumberOfColumns()),
      _nResponses(yTable.getNumberOfColumns())
{
    bool isOverflow = false;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _nBetasIntercept, _nBetasIntercept, isOverflow);
    if (isOverflow)
    {
        errorCode = ErrorCode::intOverflow;
        return;
    }
    if (!_xtx.reset(_nBetasIntercept * _nBetasIntercept))
    {
        errorCode = ErrorCode::memAllocationFailed;
        return;
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _nBetasIntercept, _nResponses, isOverflow);
    if (isOverflow)
    {
        errorCode = ErrorCode::intOverflow;
        return;
    }
    if (!_xty.reset(_nBetasIntercept * _nResponses))
    {
        errorCode = ErrorCode::memAllocationFailed;
        return;
    }
    errorCode = ErrorCode::ok;
}

template <typename algorithmFPType, CpuType cpu>
void * LinearModelReducer<algorithmFPType, cpu>::operator new(size_t size)
{
    return service_scalable_malloc<unsigned char, cpu>(size);
}

template <typename algorithmFPType, CpuType cpu>
void LinearModelReducer<algorithmFPType, cpu>::operator delete(void * p)
{
    service_scalable_free<unsigned char, cpu>((unsigned char *)p);
}

template <typename algorithmFPType, CpuType cpu>
inline algorithmFPType * LinearModelReducer<algorithmFPType, cpu>::xty()
{
    return _xty.get();
}

template <typename algorithmFPType, CpuType cpu>
inline algorithmFPType * LinearModelReducer<algorithmFPType, cpu>::xtx()
{
    return _xtx.get();
}

template <typename algorithmFPType, CpuType cpu>
inline const algorithmFPType * LinearModelReducer<algorithmFPType, cpu>::xty() const
{
    return _xty.get();
}

template <typename algorithmFPType, CpuType cpu>
inline const algorithmFPType * LinearModelReducer<algorithmFPType, cpu>::xtx() const
{
    return _xtx.get();
}

template <typename algorithmFPType, CpuType cpu>
ReducerUniquePtr LinearModelReducer<algorithmFPType, cpu>::create() const
{
    return daal::internal::makeUnique<LinearModelReducer<algorithmFPType, cpu>, DAAL_BASE_CPU>(_xTable, _yTable, _nBetasIntercept, _numRowsInBlock,
                                                                                               _numBlocks);
}

template <typename algorithmFPType, CpuType cpu>
void LinearModelReducer<algorithmFPType, cpu>::update(size_t begin, size_t end)
{
    /* SYRK and GEMM parameters */
    char up             = 'U';
    char trans          = 'T';
    char notrans        = 'N';
    algorithmFPType one = 1.0;
    DAAL_PROFILER_THREADING_TASK(reducer.update);
    if (errorCode != ErrorCode::ok)
    {
        return;
    }
    algorithmFPType * xtxPtr = xtx();
    algorithmFPType * xtyPtr = xty();

    if (!xtxPtr || !xtyPtr)
    {
        errorCode = ErrorCode::memAllocationFailed;
        return;
    }

    bool isOverflow = false;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _numRowsInBlock, _nFeatures, isOverflow);
    if (isOverflow)
    {
        errorCode = ErrorCode::intOverflow;
        return;
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _numRowsInBlock, _nResponses, isOverflow);
    if (isOverflow)
    {
        errorCode = ErrorCode::intOverflow;
        return;
    }
    const size_t numRowsInLastBlock = _numRowsInBlock + _xTable.getNumberOfRows() - _numBlocks * _numRowsInBlock;

    /// Process blocks of the input data table
    for (size_t iBlock = begin; iBlock < end; ++iBlock)
    {
        size_t nRows    = ((iBlock + 1 < _numBlocks) ? _numRowsInBlock : numRowsInLastBlock);
        size_t startRow = iBlock * _numRowsInBlock;

        ReadRows<algorithmFPType, cpu, NumericTable> xTableBD(const_cast<NumericTable &>(_xTable), startRow, nRows);
        ReadRows<algorithmFPType, cpu, NumericTable> yTableBD(const_cast<NumericTable &>(_yTable), startRow, nRows);
        if (!xTableBD.get() || !yTableBD.get())
        {
            errorCode = ErrorCode::memAllocationFailed;
            return;
        }
        algorithmFPType * xBlock = const_cast<algorithmFPType *>(xTableBD.get());
        algorithmFPType * yBlock = const_cast<algorithmFPType *>(yTableBD.get());

        /// Update the cross-product matrix with the data from the block
        {
            DAAL_PROFILER_THREADING_TASK(reducer.update.syrkXX);
            BlasInst<algorithmFPType, cpu>::xsyrk(&up, &notrans, &_nFeatures, reinterpret_cast<DAAL_INT *>(&nRows), &one, xBlock, &_nFeatures, &one,
                                                  xtxPtr, &_nBetasIntercept);
        }

        if (_nFeatures < _nBetasIntercept)
        {
            // TODO: Substitute this part with a call to gemv or reduce by column/row primitive
            DAAL_PROFILER_THREADING_TASK(reducer.update.gemm1X);
            algorithmFPType * xtxLastRowPtr = xtxPtr + _nFeatures * _nBetasIntercept;
            const algorithmFPType * xPtr    = xBlock;

            for (DAAL_INT i = 0; i < nRows; i++, xPtr += _nFeatures)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (DAAL_INT j = 0; j < _nFeatures; j++)
                {
                    xtxLastRowPtr[j] += xPtr[j];
                }
            }
            xtxLastRowPtr[_nFeatures] += algorithmFPType(nRows);
        }
        {
            DAAL_PROFILER_THREADING_TASK(reducer.update.gemmXY);
            BlasInst<algorithmFPType, cpu>::xxgemm(&notrans, &trans, &_nFeatures, &_nResponses, reinterpret_cast<DAAL_INT *>(&nRows), &one, xBlock,
                                                   &_nFeatures, yBlock, &_nResponses, &one, xtyPtr, &_nBetasIntercept);
        }
        if (_nFeatures < _nBetasIntercept)
        {
            // TODO: Substitute this part with call to gemv or reduce by column/row primitive
            DAAL_PROFILER_THREADING_TASK(reducer.update.gemm1Y);
            const algorithmFPType * yPtr = yBlock;
            for (DAAL_INT i = 0; i < nRows; i++, yPtr += _nResponses)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (DAAL_INT j = 0; j < _nResponses; j++)
                {
                    xtyPtr[j * _nBetasIntercept + _nFeatures] += yPtr[j];
                }
            }
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void LinearModelReducer<algorithmFPType, cpu>::join(daal::Reducer * otherReducer)
{
    if (errorCode != ErrorCode::ok)
    {
        return;
    }
    DAAL_PROFILER_THREADING_TASK(reducer.join);
    LinearModelReducer<algorithmFPType, cpu> * other = dynamic_cast<LinearModelReducer<algorithmFPType, cpu> *>(otherReducer);
    if (!other)
    {
        errorCode = ErrorCode::badCast;
        return;
    }
    if (other->errorCode != ErrorCode::ok)
    {
        errorCode = other->errorCode;
        return;
    }
    const algorithmFPType * otherXTX = other->xtx();
    algorithmFPType * thisXTX        = xtx();

    const algorithmFPType * otherXTY = other->xty();
    algorithmFPType * thisXTY        = xty();

    if (!otherXTX || !otherXTY || !thisXTX || !thisXTY)
    {
        errorCode = ErrorCode::memAllocationFailed;
        return;
    }
    /// It is safe to use aligned loads and stores because the data in TArrayScalableCalloc data structures is aligned
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    PRAGMA_VECTOR_ALIGNED
    for (size_t i = 0; i < (_nBetasIntercept * _nBetasIntercept); i++)
    {
        thisXTX[i] += otherXTX[i];
    }

    /// It is safe to use aligned loads and stores because the data in TArrayScalableCalloc data structures is aligned
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    PRAGMA_VECTOR_ALIGNED
    for (size_t i = 0; i < (_nBetasIntercept * _nResponses); i++)
    {
        thisXTY[i] += otherXTY[i];
    }
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
    size_t grainSize    = 1; // minimal number of data blocks to be processed by a thread
    if (hyperparameter != nullptr)
    {
        DAAL_INT64 nRowsInBlockInt64 = 0l;
        services::Status status      = hyperparameter->find(denseUpdateStepBlockSize, nRowsInBlockInt64);
        DAAL_CHECK(0l < nRowsInBlockInt64, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);
        nRowsInBlock = static_cast<size_t>(nRowsInBlockInt64);

        DAAL_INT64 grainSizeInt64 = 0l;
        status                    = hyperparameter->find(denseGrainSize, grainSizeInt64);
        DAAL_CHECK(0l < grainSizeInt64, services::ErrorIncorrectDataRange);
        DAAL_CHECK_STATUS_VAR(status);
        grainSize = static_cast<size_t>(grainSizeInt64);
    }

    size_t nBlocks = nRows / nRowsInBlock;
    nBlocks += bool(nRows % nRowsInBlock);

    LinearModelReducer<algorithmFPType, cpu> result(xTable, yTable, nBetasIntercept, nRowsInBlock, nBlocks);
    if (!result.xtx() || !result.xty())
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }
    /* Reduce input matrices X, Y into cross product X^tX and matrix-matrix product X^tY */
    daal::static_threader_reduce(nBlocks, grainSize, result);

    if (result.errorCode != LinearModelReducer<algorithmFPType, cpu>::ok)
    {
        if (result.errorCode == LinearModelReducer<algorithmFPType, cpu>::memAllocationFailed)
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        if (result.errorCode == LinearModelReducer<algorithmFPType, cpu>::intOverflow)
        {
            return services::Status(services::ErrorBufferSizeIntegerOverflow);
        }
        if (result.errorCode == LinearModelReducer<algorithmFPType, cpu>::badCast)
        {
            return services::Status(services::ErrorLinearRegressionInternal);
        }
    }

    const algorithmFPType * resultXTX = result.xtx();
    const algorithmFPType * resultXTY = result.xty();

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < (nBetasIntercept * nBetasIntercept); i++)
    {
        xtx[i] += resultXTX[i];
    }

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < (nBetasIntercept * nResponses); i++)
    {
        xty[i] += resultXTY[i];
    }

    return services::Status();
}

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
