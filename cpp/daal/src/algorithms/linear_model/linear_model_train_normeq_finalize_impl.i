/* file: linear_model_train_normeq_finalize_impl.i */
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

#include "src/algorithms/service_kernel_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"
#include "src/externals/service_stat.h"
#include "src/services/service_environment.h"
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
Status FinalizeKernel<algorithmFPType, cpu>::compute(const NumericTable & xtxTable, const NumericTable & xtyTable, NumericTable & xtxFinalTable,
                                                     NumericTable & xtyFinalTable, NumericTable & betaTable, bool interceptFlag,
                                                     const KernelHelperIface<algorithmFPType, cpu> & helper)
{
    return FinalizeKernel<algorithmFPType, cpu>::compute(xtxTable, xtyTable, xtxFinalTable, xtyFinalTable, betaTable, interceptFlag, helper, nullptr);
}

template <typename algorithmFPType, CpuType cpu>
Status FinalizeKernel<algorithmFPType, cpu>::compute(const NumericTable & xtxTable, const NumericTable & xtyTable, NumericTable & xtxFinalTable,
                                                     NumericTable & xtyFinalTable, NumericTable & betaTable, bool interceptFlag,
                                                     const KernelHelperIface<algorithmFPType, cpu> & helper,
                                                     const HyperparameterType * hyperparameter)
{
    DAAL_PROFILER_TASK(computeFinalize);
    const size_t nBetas(betaTable.getNumberOfColumns());
    const size_t nResponses(betaTable.getNumberOfRows());
    const size_t nBetasIntercept = (interceptFlag ? nBetas : (nBetas - 1));

    const size_t xtxSizeInBytes(sizeof(algorithmFPType) * nBetasIntercept * nBetasIntercept);
    const size_t xtySizeInBytes(sizeof(algorithmFPType) * nBetasIntercept * nResponses);

    TArray<algorithmFPType, cpu> betaBufferArray;
    algorithmFPType * betaBuffer(nullptr);
    Status st;
    {
        ReadRowsType xtxBlock(const_cast<NumericTable &>(xtxTable), 0, nBetasIntercept);
        DAAL_CHECK_BLOCK_STATUS(xtxBlock);
        algorithmFPType * xtx = const_cast<algorithmFPType *>(xtxBlock.get());

        if (&xtxTable != &xtxFinalTable)
        {
            DAAL_PROFILER_TASK(computeFinalize.copyToxtxFinalTable);
            DAAL_CHECK_STATUS(st, copyDataToTable(xtx, xtxSizeInBytes, xtxFinalTable));
        }

        {
            ReadRowsType xtyBlock(const_cast<NumericTable &>(xtyTable), 0, nResponses);
            DAAL_CHECK_BLOCK_STATUS(xtyBlock);
            algorithmFPType * xty = const_cast<algorithmFPType *>(xtyBlock.get());

            if (&xtyTable != &xtyFinalTable)
            {
                DAAL_PROFILER_TASK(computeFinalize.copyToxtyFinalTable);
                DAAL_CHECK_STATUS(st, copyDataToTable(xty, xtySizeInBytes, xtyFinalTable));
            }

            betaBufferArray.reset(nResponses * nBetasIntercept);
            betaBuffer = betaBufferArray.get();
            DAAL_CHECK_MALLOC(betaBuffer);

            DAAL_PROFILER_TASK(computeFinalize.betaBufCopy);
            int result = daal::services::internal::daal_memcpy_s(betaBuffer, xtySizeInBytes, xty, xtySizeInBytes);
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
        }
        {
            TArray<algorithmFPType, cpu> xtxCopyArray(nBetasIntercept * nBetasIntercept);
            algorithmFPType * xtxCopy = xtxCopyArray.get();
            DAAL_CHECK_MALLOC(xtxCopy);

            {
                DAAL_PROFILER_TASK(computeFinalize.xtxCopy);
                int result = daal::services::internal::daal_memcpy_s(xtxCopy, xtxSizeInBytes, xtx, xtxSizeInBytes);
                DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            }

            {
                DAAL_PROFILER_TASK(computeFinalize.computeBetasImpl);
                DAAL_CHECK_STATUS(st, helper.computeBetasImpl(nBetasIntercept, xtx, xtxCopy, nResponses, betaBuffer, interceptFlag, false));
            }
        }
    }

    WriteOnlyRowsType betaBlock(betaTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(betaBlock);
    algorithmFPType * beta = betaBlock.get();

    DAAL_PROFILER_TASK(computeFinalize.copyBetaToResult);
    if (nBetasIntercept == nBetas)
    {
        for (size_t i = 0; i < nResponses; i++)
        {
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 1; j < nBetas; j++)
            {
                beta[i * nBetas + j] = betaBuffer[i * nBetas + j - 1];
            }
            beta[i * nBetas] = betaBuffer[i * nBetas + nBetas - 1];
        }
    }
    else
    {
        for (size_t i = 0; i < nResponses; i++)
        {
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nBetas - 1; j++)
            {
                beta[i * nBetas + j + 1] = betaBuffer[i * nBetasIntercept + j];
            }
            beta[i * nBetas] = 0.0;
        }
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status FinalizeKernel<algorithmFPType, cpu>::copyDataToTable(const algorithmFPType * data, size_t dataSizeInBytes, NumericTable & table)
{
    WriteOnlyRowsType block(table, 0, table.getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(block);
    algorithmFPType * dst = block.get();

    int result = daal::services::internal::daal_memcpy_s(dst, dataSizeInBytes, data, dataSizeInBytes);
    return (!result) ? Status() : Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
Status FinalizeKernel<algorithmFPType, cpu>::solveSystem(DAAL_INT p, algorithmFPType * a, DAAL_INT ny, algorithmFPType * b,
                                                         const ErrorID & internalError, bool isDualSystem)
{
    /* The rounding noise in the eigenvalues of the centered X * X^t is about p * eps * lambda_max */
    const algorithmFPType cutoffFactor = isDualSystem ? static_cast<algorithmFPType>(p) : algorithmFPType(1);
    if (daal::algorithms::internal::solveSymmetricEquationsSystem<algorithmFPType, cpu>(a, b, p, ny, false, cutoffFactor))
    {
        return Status();
    }
    else
    {
        return Status(internalError);
    }
}

/**
 * Solves regression problem through the dual system: (X * X^t + lambda * I) * dualCoef = y.
 *
 * The resulting coefficients beta = X^t * dualCoef.
 *
 * The intercept case is solved for centered X and y so that the intercept is not penalized.
 *
 * @param[in]   xTable          Input data table with features of size nRows x nFeatures.
 * @param[in]   yTable          Input data table with responses of size nRows x nResponses.
 * @param[out]  betaTable       Output data table for coefficients of size nResponses x (nFeatures + 1).
 *                              The first column corresponds to the intercept (if interceptFlag is true) or zero otherwise.
 * @param[out]  xxtTable        Data table for storing X * X^t of size nRows x nRows.
 * @param[out]  dualCoefTable   Data table for storing dual coefficients of size nResponses x nRows.
 * @param[in]  interceptFlag    Flag indicating whether to include an intercept term.
 * @param[in]  helper           Helper object containing ridge parameter if needed and other kernel-related information.
 *
 * @return Status indicating success or failure of the operation.
 */
template <typename algorithmFPType, CpuType cpu>
Status DualKernel<algorithmFPType, cpu>::compute(const NumericTable & xTable, const NumericTable & yTable, NumericTable & betaTable,
                                                 NumericTable & xxtTable, NumericTable & dualCoefTable, bool interceptFlag,
                                                 const KernelHelperIface<algorithmFPType, cpu> & helper)
{
    DAAL_PROFILER_TASK(computeDual);
    const DAAL_INT nRows(xTable.getNumberOfRows());
    const DAAL_INT nFeatures(xTable.getNumberOfColumns());
    const DAAL_INT nResponses(yTable.getNumberOfColumns());
    const DAAL_INT nBetas(betaTable.getNumberOfColumns());

    DAAL_ASSERT(nBetas == nFeatures + 1);
    DAAL_CHECK(nRows > 0, services::ErrorIncorrectNumberOfObservations);

    ReadRowsType yBlock(const_cast<NumericTable &>(yTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(yBlock);
    const algorithmFPType * y = yBlock.get();

    const algorithmFPType zero(0);

    if (nRows == 1 && interceptFlag)
    {
        /* Special case: only one observation and intercept is included.
           The intercept is set to the response value and all other coefficients are zero. */
        WriteOnlyRows<algorithmFPType, cpu> betaBlock(betaTable, 0, nResponses);
        DAAL_CHECK_BLOCK_STATUS(betaBlock);
        algorithmFPType * beta = betaBlock.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(beta, zero, nBetas * nResponses);
        for (DAAL_INT j = 0; j < nResponses; ++j)
        {
            beta[j * nBetas] = y[j];
        }
        return Status();
    }

    ReadRowsType xBlock(const_cast<NumericTable &>(xTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(xBlock);
    const algorithmFPType * x = xBlock.get();

    TArrayCalloc<algorithmFPType, cpu> xMeanArray(interceptFlag ? nFeatures : 0);
    TArrayCalloc<algorithmFPType, cpu> yMeanArray(interceptFlag ? nResponses : 0);
    /* Allocate Xc so that each row starts at an aligned address */
    const DAAL_INT nAlignedElements = static_cast<DAAL_INT>(DAAL_MALLOC_DEFAULT_ALIGNMENT / sizeof(algorithmFPType));
    const DAAL_INT ldxc             = ((nFeatures + nAlignedElements - 1) / nAlignedElements) * nAlignedElements;
    TArray<algorithmFPType, cpu> xcArray(interceptFlag ? nRows * ldxc : 0);
    TArray<algorithmFPType, cpu> ycArray(interceptFlag ? nRows * nResponses : 0);
    algorithmFPType * xMean = xMeanArray.get();
    algorithmFPType * yMean = yMeanArray.get();
    algorithmFPType * xcBuf = xcArray.get();
    algorithmFPType * ycBuf = ycArray.get();
    const algorithmFPType one(1);
    const algorithmFPType negOne(-1);
    const DAAL_INT incOne(1);
    if (interceptFlag)
    {
        DAAL_PROFILER_TASK(computeDual.computeMeans);
        DAAL_CHECK_MALLOC(xMean);
        DAAL_CHECK_MALLOC(yMean);
        DAAL_CHECK_MALLOC(xcBuf);
        DAAL_CHECK_MALLOC(ycBuf);
        if (daal::internal::StatisticsInst<algorithmFPType, cpu>::xmeansOnePass(x, nFeatures, nRows, xMean)) return helper.getInternalErrorCode();
        if (daal::internal::StatisticsInst<algorithmFPType, cpu>::xmeansOnePass(y, nResponses, nRows, yMean)) return helper.getInternalErrorCode();

        /* Center the columns of X and Y: Xc = X - 1 * xMean^t, Yc = Y - 1 * yMean^t. */
        const size_t rowSizeInBytes = (static_cast<size_t>(nFeatures) + ldxc + 2 * nResponses) * sizeof(algorithmFPType);
        const size_t nRowsInBlock   = services::internal::getNumElementsFitInMemory(services::internal::getL2CacheSize() * 0.8, rowSizeInBytes, 1);
        const size_t nRowBlocks     = (static_cast<size_t>(nRows) + nRowsInBlock - 1) / nRowsInBlock;
        daal::threader_for(nRowBlocks, nRowBlocks, [&](size_t iBlock) {
            const size_t rowStart = iBlock * nRowsInBlock;
            const size_t rowEnd   = (rowStart + nRowsInBlock < static_cast<size_t>(nRows)) ? rowStart + nRowsInBlock : nRows;

            if (nResponses == 1)
            {
                for (size_t i = rowStart; i < rowEnd; ++i)
                {
                    MathInst<algorithmFPType, cpu>::vSub(nFeatures, x + i * nFeatures, xMean, xcBuf + i * ldxc);
                }
                PRAGMA_OMP_SIMD
                for (size_t i = rowStart; i < rowEnd; ++i)
                {
                    ycBuf[i] = y[i] - yMean[0];
                }
            }
            else
            {
                for (size_t i = rowStart; i < rowEnd; ++i)
                {
                    MathInst<algorithmFPType, cpu>::vSub(nFeatures, x + i * nFeatures, xMean, xcBuf + i * ldxc);
                    MathInst<algorithmFPType, cpu>::vSub(nResponses, y + i * nResponses, yMean, ycBuf + i * nResponses);
                }
            }
        });
    }
    const DAAL_INT ldx         = interceptFlag ? ldxc : nFeatures;
    const algorithmFPType * xc = interceptFlag ? xcBuf : x;
    const algorithmFPType * yc = interceptFlag ? ycBuf : y;

    /* Compute X * X^t */
    /* Use the storage provided by the caller if it has the size of the dual problem (nRows x nRows),
       otherwise (e.g. the model's X^t * X table in the classic API) use a temporary buffer */
    const bool useXXtTable =
        (xxtTable.getNumberOfRows() == static_cast<size_t>(nRows)) && (xxtTable.getNumberOfColumns() == static_cast<size_t>(nRows));
    WriteOnlyRowsType xxtBlock;
    TArray<algorithmFPType, cpu> xxtArray;
    algorithmFPType * xxt = nullptr;
    if (useXXtTable)
    {
        xxtBlock.set(xxtTable, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(xxtBlock);
        xxt = xxtBlock.get();
    }
    else
    {
        xxt = xxtArray.reset(nRows * nRows);
        DAAL_CHECK_MALLOC(xxt);
    }

    {
        DAAL_PROFILER_TASK(computeDual.computeXXt);
        /* Compute X * X^t using GEMM as GEMM is faster than SYRK with TBB threading */
        BlasInst<algorithmFPType, cpu>::xgemm("T", "N", &nRows, &nRows, &nFeatures, &one, xc, &ldx, xc, &ldx, &zero, xxt, &nRows);
    }

    /* Form right hand side, the solver works on dualCoef of size nResponses x nRows,
       where i-th row represents the dual coefficients for the i-th observation */
    const bool useDualCoefTable =
        (dualCoefTable.getNumberOfRows() == static_cast<size_t>(nResponses)) && (dualCoefTable.getNumberOfColumns() == static_cast<size_t>(nRows));
    WriteOnlyRowsType dualCoefBlock;
    TArray<algorithmFPType, cpu> dualCoefArray;
    algorithmFPType * dualCoef = nullptr;
    if (useDualCoefTable)
    {
        /* If dualCoefTable comes from oneDAL API, use it to store the dual coefficients directly. */
        dualCoefBlock.set(dualCoefTable, 0, nResponses);
        DAAL_CHECK_BLOCK_STATUS(dualCoefBlock);
        dualCoef = dualCoefBlock.get();
    }
    else
    {
        /* If dualCoefTable comes from legacy DAAL API, allocate memory for storing the dual coefficients. */
        dualCoef = dualCoefArray.reset(nRows * nResponses);
        DAAL_CHECK_MALLOC(dualCoef);
    }
    {
        DAAL_PROFILER_TASK(computeDual.formRightHandSide);
        /* dualCoef = Yc^t */
        BlasInst<algorithmFPType, cpu>::xomatcopy("R", "T", &nRows, &nResponses, &one, yc, &nResponses, dualCoef, &nRows);
    }

    Status st;
    {
        DAAL_PROFILER_TASK(computeDual.solveSystem);
        TArray<algorithmFPType, cpu> xxtCopyArray;
        algorithmFPType * xxtCopy = xxt;
        if (nResponses > 1)
        {
            /* The case of nResponses == 1 is solved in place, as X * X^t is not used after the solve. */
            xxtCopy = xxtCopyArray.reset(nRows * nRows);
            DAAL_CHECK_MALLOC(xxtCopy);
            const size_t xxtSizeInBytes = nRows * nRows * sizeof(algorithmFPType);
            DAAL_CHECK(!daal::services::internal::daal_memcpy_s(xxtCopy, xxtSizeInBytes, xxt, xxtSizeInBytes),
                       services::ErrorMemoryCopyFailedInternal);
        }
        DAAL_CHECK_STATUS(st, helper.computeBetasImpl(nRows, xxt, xxtCopy, nResponses, dualCoef, false, true));
    }

    /* Each beta row is [intercept, coefficients] */
    WriteOnlyRows<algorithmFPType, cpu> betaBlock(betaTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(betaBlock);
    algorithmFPType * beta = betaBlock.get();
    {
        DAAL_PROFILER_TASK(computeDual.computePrimalCoefficients);

        /* Compute the coefficients beta[1..nFeatures] = Xc^t * dualCoef */
        BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &nFeatures, &nResponses, &nRows, &one, xc, &ldx, dualCoef, &nRows, &zero, beta + 1, &nBetas);
    }

    /* Compute intercepts: yMean = yMean - xMean^t * beta[:, 1:] */
    if (interceptFlag)
    {
        /* Note: BLAS prevents in-place computation using beta as input and output, so we need to store the result in yMean explicitly */
        BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &incOne, &nResponses, &nFeatures, &negOne, xMean, &incOne, beta + 1, &nBetas, &one, yMean,
                                              &incOne);
    }
    for (DAAL_INT j = 0; j < nResponses; ++j) beta[j * nBetas] = interceptFlag ? yMean[j] : zero;
    return st;
}

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
