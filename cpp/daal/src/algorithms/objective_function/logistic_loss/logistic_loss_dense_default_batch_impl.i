/* file: logistic_loss_dense_default_batch_impl.i */
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
//  Implementation of logloss algorithm
//--
*/
#include "src/services/service_data_utils.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/externals/service_profiler.h"

#include "src/algorithms/objective_function/common/objective_function_utils.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// Logistic loss function, L(x,y,b) = -[y*ln(sigmoid(f)) + (1 - y)*ln(1-sigmoid(f))]
// where sigmoid(f) = 1/(1 + exp(-f), f = x*b
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
static void applyBetaImpl(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb, size_t nRows, size_t nCols, bool bIntercept,
                          bool bThreaded)
{
    char trans           = 'T';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;
    DAAL_INT n           = (DAAL_INT)nRows;
    DAAL_INT dim         = (DAAL_INT)nCols;
    DAAL_INT ione        = 1;
    if (bThreaded)
    {
        BlasInst<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    }
    else
    {
        BlasInst<algorithmFPType, cpu>::xxgemv(&trans, &dim, &n, &one, x, &dim, beta + 1, &ione, &zero, xb, &ione);
    }
    if (bIntercept)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i)
        {
            xb[i] += beta[0];
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void LogLossKernel<algorithmFPType, method, cpu>::applyBeta(const algorithmFPType * x, const algorithmFPType * beta, algorithmFPType * xb,
                                                            size_t nRows, size_t nCols, bool bIntercept)
{
    applyBetaImpl<algorithmFPType, cpu>(x, beta, xb, nRows, nCols, bIntercept, false);
}

template <typename algorithmFPType, CpuType cpu>
static void vexp(const algorithmFPType * f, algorithmFPType * exp, size_t n)
{
    const algorithmFPType expThreshold = daal::internal::MathInst<algorithmFPType, cpu>::vExpThreshold();
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i)
    {
        exp[i] = -f[i];
        /* make all values less than threshold as threshold value
        to fix slow work on vExp on large negative inputs */
        if (exp[i] < expThreshold) exp[i] = expThreshold;
    }
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(n, exp, exp);
}

template <typename algorithmFPType, CpuType cpu>
static void sigmoids(algorithmFPType * exp, size_t n, size_t offset)
{
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i)
    {
        const auto sigm = static_cast<algorithmFPType>(1.0) / (static_cast<algorithmFPType>(1.0) + exp[i]);
        exp[i]          = sigm;
        exp[i + offset] = 1 - sigm;
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void LogLossKernel<algorithmFPType, method, cpu>::sigmoid(const algorithmFPType * f, algorithmFPType * s, size_t n)
{
    //s = exp(-f)
    vexp<algorithmFPType, cpu>(f, s, n);
    //s = sigm(f)
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < n; ++i)
    {
        s[i] = static_cast<algorithmFPType>(1.0) / (static_cast<algorithmFPType>(1.0) + s[i]);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LogLossKernel<algorithmFPType, method, cpu>::doCompute(const NumericTable * dataNT, const NumericTable * dependentVariablesNT,
                                                                        size_t n, size_t p, NumericTable * betaNT, NumericTable * valueNT,
                                                                        NumericTable * hessianNT, NumericTable * gradientNT,
                                                                        NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                                                                        NumericTable * lipschitzConstant, Parameter * parameter)
{
    SafeStatus safeStat;
    const size_t nBeta = p + 1;
    DAAL_ASSERT(betaNT->getNumberOfColumns() == 1);
    DAAL_ASSERT(betaNT->getNumberOfRows() == nBeta);

    const algorithmFPType * b;
    HomogenNumericTable<algorithmFPType> * hmgBeta = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(betaNT);
    ReadRows<algorithmFPType, cpu> betar;

    if (hmgBeta)
    {
        b = (*hmgBeta).getArray();
    }
    else
    {
        betar.set(betaNT, 0, nBeta);
        DAAL_CHECK_BLOCK_STATUS(betar);
        b = betar.get();
    }

    if (proximalProjection)
    {
        DAAL_ASSERT(proximalProjection->getNumberOfRows() == nBeta);
        algorithmFPType * prox;

        HomogenNumericTable<algorithmFPType> * hmgProx = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(proximalProjection);
        WriteRows<algorithmFPType, cpu> pr;
        if (hmgProx)
        {
            prox = hmgProx->getArray();
        }
        else
        {
            pr.set(proximalProjection, 0, nBeta);
            DAAL_CHECK_BLOCK_STATUS(pr);
            prox = pr.get();
        }

        prox[0] = b[0];
        for (size_t i = 1; i < nBeta; i++)
        {
            if (b[i] > parameter->penaltyL1)
            {
                prox[i] = b[i] - parameter->penaltyL1;
            }
            if (b[i] < -parameter->penaltyL1)
            {
                prox[i] = b[i] + parameter->penaltyL1;
            }
            if (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(b[i]) <= parameter->penaltyL1)
            {
                prox[i] = 0;
            }
        }
    }

    if (lipschitzConstant)
    {
        DAAL_ASSERT(lipschitzConstant->getNumberOfRows() == 1);
        WriteRows<algorithmFPType, cpu> lipschitzConstantPtr(lipschitzConstant, 0, 1);
        algorithmFPType & c = *lipschitzConstantPtr.get();

        const size_t blockSize        = 256;
        const size_t nBlocks          = n / blockSize + !!(n % blockSize);
        algorithmFPType globalMaxNorm = 0;

        TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(lipschitzConstant->getNumberOfRows());

        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            algorithmFPType & _maxNorm = *tlsData.local();
            const size_t startRow      = iBlock * blockSize;
            const size_t finishRow     = (iBlock + 1 == nBlocks ? n : (iBlock + 1) * blockSize);
            ReadRows<algorithmFPType, cpu> xr(const_cast<NumericTable *>(dataNT), startRow, finishRow - startRow);
            DAAL_CHECK_BLOCK_STATUS_THR(xr);
            const algorithmFPType * const x = xr.get();
            algorithmFPType curentNorm      = 0;
            for (size_t i = 0; i < finishRow - startRow; i++)
            {
                curentNorm = 0;
                for (size_t j = 0; j < p; j++)
                {
                    curentNorm += x[i * p + j] * x[i * p + j];
                }
                if (curentNorm > _maxNorm)
                {
                    _maxNorm = curentNorm;
                }
            }
        });
        tlsData.reduce([&](algorithmFPType * maxNorm) {
            if (globalMaxNorm < *maxNorm)
            {
                globalMaxNorm = *maxNorm;
            }
        });

        algorithmFPType alpha_scaled = algorithmFPType(parameter->penaltyL2) / algorithmFPType(n);
        algorithmFPType lipschitz    = 0.25 * (globalMaxNorm + algorithmFPType(parameter->interceptFlag)) + alpha_scaled;
        algorithmFPType displacement = daal::internal::MathInst<algorithmFPType, cpu>::sMin(2 * parameter->penaltyL2, lipschitz);
        c                            = 2 * lipschitz + displacement;
    }

    algorithmFPType nonSmoothTerm = 0;
    if (nonSmoothTermValue)
    {
        WriteRows<algorithmFPType, cpu> vr(nonSmoothTermValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType & v = *vr.get();

        if ((parameter->penaltyL1 > 0))
        {
            for (size_t i = 1; i < nBeta; ++i)
            {
                nonSmoothTerm += (b[i] < 0 ? -b[i] : b[i]) * parameter->penaltyL1;
            }
        }
        v = nonSmoothTerm;
    }

    if (valueNT || gradientNT || hessianNT)
    {
        TNArray<algorithmFPType, 16, cpu> f;
        TNArray<algorithmFPType, 32, cpu> sg;

        TArrayScalable<algorithmFPType, cpu> fScalable;
        TArrayScalable<algorithmFPType, cpu> sgScalable;

        algorithmFPType * fPtr;
        algorithmFPType * sgPtr;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, 2);
        if (n < 16)
        {
            f.reset(n);
            sg.reset(2 * n);
            fPtr  = f.get();
            sgPtr = sg.get();
        }
        else
        {
            fScalable.reset(n);
            sgScalable.reset(2 * n);
            fPtr  = fScalable.get();
            sgPtr = sgScalable.get();
        }
        DAAL_CHECK(fPtr && sgPtr, ErrorMemoryAllocationFailed);

        const bool bL1 = parameter->penaltyL1 > 0;
        const bool bL2 = parameter->penaltyL2 > 0;

        const size_t iFirstBeta   = parameter->interceptFlag ? 0 : 1;
        const algorithmFPType div = static_cast<algorithmFPType>(1) / static_cast<algorithmFPType>(n);

        const size_t nRowsInBlock = 512;
        const size_t nDataBlocks  = n / nRowsInBlock + !!(n % nRowsInBlock);

        TlsMem<algorithmFPType, cpu> tlsData(2 * nRowsInBlock);

        TArrayScalable<algorithmFPType, cpu> values;
        if (valueNT)
        {
            values.reset(nDataBlocks);
            DAAL_CHECK_MALLOC(values.get());
        }
        TArrayScalable<algorithmFPType, cpu> grads;
        if (gradientNT)
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nDataBlocks, p);
            grads.reset(nDataBlocks * p);
            DAAL_CHECK_MALLOC(grads.get());
        }

        TArrayScalable<algorithmFPType, cpu> interceptGrad;
        if (gradientNT && parameter->interceptFlag)
        {
            interceptGrad.reset(nDataBlocks);
            DAAL_CHECK_MALLOC(interceptGrad.get());
        }

        daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? n - iBlock * nRowsInBlock : nRowsInBlock;

            ReadRows<algorithmFPType, cpu> xr(const_cast<NumericTable *>(dataNT), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xr);
            ReadRows<algorithmFPType, cpu> yr(const_cast<NumericTable *>(dependentVariablesNT), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(yr);
            const algorithmFPType * const xLocal = xr.get();
            const algorithmFPType * const yLocal = yr.get();

            algorithmFPType * const fPtrLocal  = fPtr + iStartRow;
            algorithmFPType * const sgPtrLocal = sgPtr + iStartRow;

            //f = X*b + b0
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(applyBeta);
                applyBeta(xLocal, b, fPtrLocal, nRowsToProcess, p, parameter->interceptFlag);
            }

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(sigmoids);
                //s = exp(-f)
                vexp<algorithmFPType, cpu>(fPtrLocal, sgPtrLocal, nRowsToProcess);

                //s = sigm(f), s1 = 1 - s
                sigmoids<algorithmFPType, cpu>(sgPtrLocal, nRowsToProcess, n);
            }

            if (valueNT)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(logLoss.computeValueResult);
                algorithmFPType * const ls = tlsData.local();
                DAAL_CHECK_THR(ls, services::ErrorMemoryAllocationFailed);
                algorithmFPType * const ls1 = ls + nRowsInBlock;

                daal::internal::MathInst<algorithmFPType, cpu>::vLog(nRowsToProcess, sgPtrLocal, ls);
                daal::internal::MathInst<algorithmFPType, cpu>::vLog(nRowsToProcess, sgPtrLocal + n, ls1);

                algorithmFPType localValue(0);

                for (size_t i = 0; i < nRowsToProcess; ++i)
                {
                    localValue += yLocal[i] * ls[i] + (static_cast<algorithmFPType>(1) - yLocal[i]) * ls1[i];
                }
                localValue *= -div;

                values[iBlock] = localValue;
            }

            if (gradientNT)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);
                DAAL_ASSERT(gradientNT->getNumberOfRows() == nBeta);

                const char notrans         = 'N';
                const algorithmFPType one  = 1.0;
                const algorithmFPType zero = 0.0;
                const DAAL_INT yDim        = 1;
                DAAL_ASSERT(p <= services::internal::MaxVal<DAAL_INT>::get());
                const DAAL_INT dim = static_cast<DAAL_INT>(p);
                DAAL_ASSERT(nRowsToProcess <= services::internal::MaxVal<DAAL_INT>::get());
                const DAAL_INT nN          = static_cast<DAAL_INT>(nRowsToProcess);
                algorithmFPType * const pg = grads.get() + iBlock * p;

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nRowsToProcess; ++i)
                {
                    sgPtrLocal[i] -= yLocal[i];
                }

                daal::internal::BlasInst<algorithmFPType, cpu>::xxgemm(&notrans, &notrans, &dim, &yDim, &nN, &one, xLocal, &dim, sgPtrLocal, &nN,
                                                                       &zero, pg, &dim);

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nRowsToProcess; ++i)
                {
                    sgPtrLocal[i] += yLocal[i];
                }

                if (parameter->interceptFlag)
                {
                    algorithmFPType interceptGradLocal(0);

                    for (size_t i = 0; i < nRowsToProcess; ++i)
                    {
                        interceptGradLocal += (sgPtrLocal[i] - yLocal[i]);
                    }

                    interceptGrad[iBlock] = interceptGradLocal;
                }
            }
        });

        if (valueNT)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(logLoss.computeValueResult);

            WriteRows<algorithmFPType, cpu> vr(valueNT, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(vr);
            algorithmFPType & value = *vr.get();
            value                   = 0;

            for (size_t i = 0; i < nDataBlocks; ++i)
            {
                value += values[i];
            }

            if (bL2)
            {
                for (size_t i = 1; i < nBeta; ++i)
                {
                    value += b[i] * b[i] * parameter->penaltyL2;
                }
            }

            if (bL1)
            {
                if (nonSmoothTermValue)
                {
                    value += nonSmoothTerm;
                }
                else
                {
                    for (size_t i = 1; i < nBeta; ++i)
                    {
                        value += (b[i] < 0 ? -b[i] : b[i]) * parameter->penaltyL1;
                    }
                }
            }
        }

        if (gradientNT)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(applyGradient);
            algorithmFPType * g;
            HomogenNumericTable<algorithmFPType> * const hmgGrad = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(gradientNT);
            WriteRows<algorithmFPType, cpu> gr;
            if (hmgGrad)
            {
                g = hmgGrad->getArray();
            }
            else
            {
                gr.set(gradientNT, 0, nBeta);
                DAAL_CHECK_BLOCK_STATUS(gr);
                g = gr.get();
            }

            const algorithmFPType * const gradsPtr         = grads.get();
            const algorithmFPType * const interceptGradPtr = interceptGrad.get();

            int result = services::internal::daal_memcpy_s(g + 1, p * sizeof(algorithmFPType), gradsPtr, p * sizeof(algorithmFPType));
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

            for (size_t i = 1; i < nDataBlocks; i++)
            {
                for (size_t j = 0; j < p; j++)
                {
                    g[j + 1] += gradsPtr[i * p + j];
                }
            }

            g[0] = 0;
            if (parameter->interceptFlag)
            {
                for (size_t i = 0; i < nDataBlocks; i++)
                {
                    g[0] += interceptGradPtr[i];
                }
            }
            for (size_t i = iFirstBeta; i < nBeta; ++i)
            {
                g[i] *= div;
            }

            if (bL2)
            {
                for (size_t i = 1; i < nBeta; ++i)
                {
                    g[i] += 2. * b[i] * parameter->penaltyL2;
                }
            }
        }

        if (hessianNT)
        {
            ReadRows<algorithmFPType, cpu> xr(const_cast<NumericTable *>(dataNT), 0, n);
            DAAL_CHECK_BLOCK_STATUS(xr);
            const algorithmFPType * const x = xr.get();
            DAAL_ASSERT(hessianNT->getNumberOfRows() == nBeta);
            WriteRows<algorithmFPType, cpu> hr(hessianNT, 0, nBeta * nBeta);
            DAAL_CHECK_BLOCK_STATUS(hr);
            algorithmFPType * h = hr.get();

            algorithmFPType * s = sgPtr;
            for (size_t i = 0; i < n; ++i)
            {
                s[i] *= s[i + n]; //sigmoid derivative at x[i]
            }

            h[0] = 0;
            if (parameter->interceptFlag)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    h[0] += s[i];
                }
                h[0] *= div; //average of sigmoid derivatives

                //first row and column
                for (size_t k = 1; k < nBeta; ++k)
                {
                    algorithmFPType val = 0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        val += s[i] * x[i * p + k - 1];
                    }
                    h[k]         = val * div;
                    h[k * nBeta] = val * div;
                }
            }
            else
            {
                //first row and column
                for (size_t k = 1; k < nBeta; ++k)
                {
                    h[k]         = 0;
                    h[k * nBeta] = 0;
                }
            }
            //rows 1,..
            for (size_t j = 1; j < nBeta; ++j)
            {
                for (size_t k = j; k < nBeta; ++k)
                {
                    algorithmFPType val = 0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        val += x[i * p + j - 1] * x[i * p + k - 1] * s[i];
                    }
                    h[j * nBeta + k] = val * div;
                    h[k * nBeta + j] = val * div;
                }
                h[j * nBeta + j] += 2. * parameter->penaltyL2;
            }
        }
        DAAL_CHECK_SAFE_STATUS()
    }
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LogLossKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataNT, NumericTable * dependentVariablesNT,
                                                                      NumericTable * betaNT, NumericTable * valueNT, NumericTable * hessianNT,
                                                                      NumericTable * gradientNT, NumericTable * nonSmoothTermValue,
                                                                      NumericTable * proximalProjection, NumericTable * lipschitzConstant,
                                                                      Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LogLossKernel.compute);

    const size_t nRows                                = dataNT->getNumberOfRows();
    const daal::data_management::NumericTable * ntInd = parameter->batchIndices.get();
    if (ntInd && (ntInd->getNumberOfColumns() == nRows)) ntInd = nullptr;

    const size_t p = dataNT->getNumberOfColumns();
    if (ntInd)
    {
        const size_t n = ntInd->getNumberOfColumns();
        services::Status s;
        HomogenNumericTable<algorithmFPType> * hmgData               = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(dataNT);
        HomogenNumericTable<algorithmFPType> * hmgDependentVariables = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(dependentVariablesNT);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, p);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * p, sizeof(algorithmFPType));

        if (_aX.size() < n * p)
        {
            _aX.reset(n * p);
            DAAL_CHECK_MALLOC(_aX.get());
        }
        if (_aY.size() < n)
        {
            _aY.reset(n);
            DAAL_CHECK_MALLOC(_aY.get());
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(getXY);
            s |= objective_function::internal::getXY<algorithmFPType, cpu>(dataNT, dependentVariablesNT, ntInd, _aX.get(), _aY.get(), nRows, n, p);
        }
        auto internalDataNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_aX.get(), p, n);
        DAAL_CHECK_MALLOC(internalDataNT.get());
        auto internalDependentVariablesNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_aY.get(), 1, n);
        DAAL_CHECK_MALLOC(internalDependentVariablesNT.get());
        s |= doCompute(internalDataNT.get(), internalDependentVariablesNT.get(), n, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue,
                       proximalProjection, lipschitzConstant, parameter);
        return s;
    }
    return doCompute(dataNT, dependentVariablesNT, nRows, p, betaNT, valueNT, hessianNT, gradientNT, nonSmoothTermValue, proximalProjection,
                     lipschitzConstant, parameter);
}

} // namespace internal

} // namespace logistic_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
