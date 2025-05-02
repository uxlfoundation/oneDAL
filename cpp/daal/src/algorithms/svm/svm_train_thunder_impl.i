/* file: svm_train_thunder_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM training algorithm implementation using Thunder method
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Zeyi Wen, Jiashuai Shi, Bingsheng He
//     ThunderSVM: A Fast SVM Library on GPUs and CPUs,
//     Journal of Machine Learning Research, 19, 1-5 (2018)
//  2. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  3. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  4. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_THUNDER_I__
#define __SVM_TRAIN_THUNDER_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_profiler.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"

#include "src/algorithms/svm/svm_train_common.h"
#include "src/algorithms/svm/svm_train_thunder_workset.h"
#include "src/algorithms/svm/svm_train_thunder_cache.h"
#include "src/algorithms/svm/svm_train_result.h"

#include "src/algorithms/svm/svm_train_common_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable,
                                                                      NumericTable & yTable, daal::algorithms::Model * r,
                                                                      const KernelParameter & svmPar)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(COMPUTE);

    services::Status status;

    const algorithmFPType C                 = svmPar.C;
    const algorithmFPType accuracyThreshold = svmPar.accuracyThreshold;
    const algorithmFPType tau               = svmPar.tau;
    const algorithmFPType epsilon           = svmPar.epsilon;
    const algorithmFPType nu                = svmPar.nu;
    const size_t maxIterations              = svmPar.maxIterations;
    const size_t cacheSize                  = svmPar.cacheSize;
    const auto kernel                       = svmPar.kernel->clone();
    const auto svmType                      = svmPar.svmType;

    const size_t nVectors = xTable->getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, 2);
    const size_t nTrainVectors = (svmType == SvmType::regression || svmType == SvmType::nu_regression) ? nVectors * 2 : nVectors;
    TArray<algorithmFPType, cpu> yTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(yTArray.get());
    algorithmFPType * const y = yTArray.get();

    TArray<algorithmFPType, cpu> gradTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(gradTArray.get());
    algorithmFPType * const grad = gradTArray.get();

    TArray<algorithmFPType, cpu> alphaTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(alphaTArray.get());
    algorithmFPType * const alpha = alphaTArray.get();

    TArray<algorithmFPType, cpu> cwTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(cwTArray.get());
    algorithmFPType * const cw = cwTArray.get();

    size_t nNonZeroWeights = nTrainVectors;
    if (svmType == SvmType::classification || svmType == SvmType::nu_classification)
    {
        DAAL_CHECK_STATUS(status, classificationInit(yTable, wTable, C, nu, y, grad, alpha, cw, nNonZeroWeights, svmType));
    }
    else
    {
        DAAL_CHECK_STATUS(status, regressionInit(yTable, wTable, C, nu, epsilon, y, grad, alpha, cw, nNonZeroWeights, svmType));
    }

    TaskWorkingSet<algorithmFPType, cpu> workSet(nNonZeroWeights, nTrainVectors, maxBlockSize, svmType);
    DAAL_CHECK_STATUS(status, workSet.init());
    const size_t nWS = workSet.getSize();

    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);
    size_t sameLocalDiff     = 0;

    TArray<algorithmFPType, cpu> buffer(nWS * MemSmoId::latest + nWS * nWS);
    DAAL_CHECK_MALLOC(buffer.get());

    TArray<algorithmFPType, cpu> deltaAlpha(nWS);
    DAAL_CHECK_MALLOC(deltaAlpha.get());

    TArray<char, cpu> I(nWS);
    DAAL_CHECK_MALLOC(I.get());

    size_t defaultCacheSize = services::internal::min<cpu, size_t>(nVectors, cacheSize / nVectors / sizeof(algorithmFPType));
    defaultCacheSize        = services::internal::max<cpu, size_t>(nWS, defaultCacheSize);
    auto cachePtr           = SVMCache<thunder, lruCache, algorithmFPType, cpu>::create(defaultCacheSize, nWS, nVectors, xTable, kernel, status);
    DAAL_CHECK_STATUS_VAR(status);

    if (svmType == SvmType::nu_classification || svmType == SvmType::nu_regression)
    {
        DAAL_CHECK_STATUS(status, initGrad(xTable, kernel, nVectors, nTrainVectors, y, alpha, grad));
    }

    size_t iter = 0;
    for (; iter < maxIterations; ++iter)
    {
        if (iter != 0)
        {
            DAAL_CHECK_STATUS(status, workSet.copyLastToFirst());
        }

        DAAL_CHECK_STATUS(status, workSet.select(y, alpha, grad, cw));
        const uint32_t * const wsIndices = workSet.getIndices();
        algorithmFPType ** kernelSOARes  = nullptr;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(getRowsBlock);

            DAAL_CHECK_STATUS(status, cachePtr->getRowsBlock(wsIndices, nWS, kernelSOARes));
        }

        DAAL_CHECK_STATUS(status, SMOBlockSolver(y, grad, wsIndices, kernelSOARes, nVectors, nWS, cw, accuracyThreshold, tau, buffer.get(), I.get(),
                                                 alpha, deltaAlpha.get(), diff, svmType));

        DAAL_CHECK_STATUS(status, updateGrad(kernelSOARes, deltaAlpha.get(), grad, nVectors, nTrainVectors, nWS));
        if (checkStopCondition(diff, diffPrev, accuracyThreshold, sameLocalDiff) && iter >= nNoChanges) break;
        diffPrev = diff;
    }

    cachePtr->clear();
    SaveResultTask<algorithmFPType, cpu> saveResult(nVectors, y, alpha, grad, svmType, cachePtr.get());
    DAAL_CHECK_STATUS(status, saveResult.compute(xTable, *static_cast<Model *>(r), cw));

    return status;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::classificationInit(NumericTable & yTable, const NumericTablePtr & wTable,
                                                                                 const algorithmFPType c, const algorithmFPType nu,
                                                                                 algorithmFPType * const y, algorithmFPType * const grad,
                                                                                 algorithmFPType * const alpha, algorithmFPType * const cw,
                                                                                 size_t & nNonZeroWeights, const SvmType svmType)
{
    services::Status status;
    const size_t nVectors = yTable.getNumberOfRows();
    /* The operation copy is lightweight, therefore a large size is chosen
        so that the number of blocks is a reasonable number. */
    const size_t blockSize = 16384;
    const size_t nBlocks   = nVectors / blockSize + !!(nVectors % blockSize);

    DAAL_ITTNOTIFY_SCOPED_TASK(init.set);
    TlsSum<size_t, cpu> weightsCounter(1);
    TlsSum<algorithmFPType, cpu> weightsSumTls(1);
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
        const size_t startRow     = iBlock * blockSize;
        const size_t nRowsInBlock = (iBlock != nBlocks - 1) ? blockSize : nVectors - iBlock * blockSize;

        ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, startRow, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(mtY);
        const algorithmFPType * const yIn = mtY.get();

        ReadColumns<algorithmFPType, cpu> mtW(wTable.get(), 0, startRow, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(mtW);
        const algorithmFPType * const weights = mtW.get();

        size_t * const wc          = weights ? weightsCounter.local() : nullptr;
        algorithmFPType * const ws = weights ? weightsSumTls.local() : nullptr;
        for (size_t i = 0; i < nRowsInBlock; ++i)
        {
            y[i + startRow]     = yIn[i] == algorithmFPType(0) ? algorithmFPType(-1) : yIn[i];
            grad[i + startRow]  = (svmType == SvmType::classification) ? -y[i + startRow] : algorithmFPType(0);
            alpha[i + startRow] = algorithmFPType(0);
            cw[i + startRow]    = weights ? weights[i] * c : c;
            if (weights)
            {
                *wc += static_cast<size_t>(weights[i] != algorithmFPType(0));
                *ws += weights[i];
            }
        }
    });

    algorithmFPType nuWeightsSum = nu * nVectors;

    if (wTable.get())
    {
        weightsCounter.reduceTo(&nNonZeroWeights, 1);
        weightsSumTls.reduceTo(&nuWeightsSum, 1);
        nuWeightsSum *= nu;
    }

    if (svmType == SvmType::nu_classification)
    {
        algorithmFPType sumPos = nuWeightsSum / algorithmFPType(2);
        algorithmFPType sumNeg = nuWeightsSum / algorithmFPType(2);

        for (size_t i = 0; i < nVectors; ++i)
        {
            if (y[i] > 0)
            {
                alpha[i] = services::internal::min<cpu, algorithmFPType>(sumPos, cw[i]);
                sumPos -= alpha[i];
            }
            else
            {
                alpha[i] = services::internal::min<cpu, algorithmFPType>(sumNeg, cw[i]);
                sumNeg -= alpha[i];
            }
        }
    }

    return status;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::regressionInit(NumericTable & yTable, const NumericTablePtr & wTable,
                                                                             const algorithmFPType c, const algorithmFPType nu,
                                                                             const algorithmFPType epsilon, algorithmFPType * const y,
                                                                             algorithmFPType * const grad, algorithmFPType * const alpha,
                                                                             algorithmFPType * const cw, size_t & nNonZeroWeights,
                                                                             const SvmType svmType)
{
    services::Status status;
    const size_t nVectors = yTable.getNumberOfRows();
    /* The operation copy is lightweight, therefore a large size is chosen
        so that the number of blocks is a reasonable number. */
    const size_t blockSize = 16384;
    const size_t nBlocks   = nVectors / blockSize + !!(nVectors % blockSize);

    TlsSum<size_t, cpu> weightsCounter(1);
    TlsSum<algorithmFPType, cpu> cwSumTls(1);
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
        const size_t startRow     = iBlock * blockSize;
        const size_t nRowsInBlock = (iBlock != nBlocks - 1) ? blockSize : nVectors - iBlock * blockSize;

        ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, startRow, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(mtY);
        const algorithmFPType * const yIn = mtY.get();

        ReadColumns<algorithmFPType, cpu> mtW(wTable.get(), 0, startRow, nRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(mtW);
        const algorithmFPType * const weights = mtW.get();

        size_t * const wc           = weights ? weightsCounter.local() : nullptr;
        algorithmFPType * const cws = weights ? cwSumTls.local() : nullptr;
        for (size_t i = 0; i < nRowsInBlock; ++i)
        {
            y[i + startRow]            = algorithmFPType(1.0);
            y[i + startRow + nVectors] = algorithmFPType(-1.0);

            grad[i + startRow]            = (svmType == SvmType::regression) ? epsilon - yIn[i] : -yIn[i];
            grad[i + startRow + nVectors] = (svmType == SvmType::regression) ? -epsilon - yIn[i] : -yIn[i];

            alpha[i + startRow]            = algorithmFPType(0);
            alpha[i + startRow + nVectors] = algorithmFPType(0);

            cw[i + startRow]            = weights ? weights[i] * c : c;
            cw[i + startRow + nVectors] = weights ? weights[i] * c : c;
            if (weights)
            {
                *wc += 2 * static_cast<size_t>(weights[i] != algorithmFPType(0));
                *cws += cw[i + startRow];
            }
        }
    });

    algorithmFPType cwSum = c * nVectors;

    if (wTable.get())
    {
        weightsCounter.reduceTo(&nNonZeroWeights, 1);
        cwSumTls.reduceTo(&cwSum, 1);
    }

    if (svmType == SvmType::nu_regression)
    {
        algorithmFPType sum = nu * cwSum / algorithmFPType(2);
        for (size_t i = 0; i < nVectors; ++i)
        {
            alpha[i] = alpha[i + nVectors] = services::internal::min<cpu, algorithmFPType>(sum, cw[i]);
            sum -= alpha[i];
        }
    }

    return status;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::SMOBlockSolver(
    const algorithmFPType * y, const algorithmFPType * grad, const uint32_t * wsIndices, algorithmFPType ** kernelWS, const size_t nVectors,
    const size_t nWS, const algorithmFPType * cw, const double accuracyThreshold, const double tau, algorithmFPType * buffer, char * I,
    algorithmFPType * alpha, algorithmFPType * deltaAlpha, algorithmFPType & localDiff, SvmType svmType) const
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SMOBlockSolver);
    services::Status status;

    const size_t innerMaxIterations(nWS * cInnerIterations);

    algorithmFPType * const alphaLocal    = buffer + nWS * MemSmoId::alphaBuffID;
    algorithmFPType * const yLocal        = buffer + nWS * MemSmoId::yBuffID;
    algorithmFPType * const gradLocal     = buffer + nWS * MemSmoId::gradBuffID;
    algorithmFPType * const kdLocal       = buffer + nWS * MemSmoId::kdBuffID;
    algorithmFPType * const oldAlphaLocal = buffer + nWS * MemSmoId::oldAlphaBuffID;
    algorithmFPType * const cwLocal       = buffer + nWS * MemSmoId::cwBuffID;
    algorithmFPType * const kernelLocal   = buffer + nWS * MemSmoId::latest;

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(SMOBlockSolver.init);
        SafeStatus safeStat;

        /* Gather data to local buffers */
        const size_t blockSizeWS = services::internal::min<cpu, algorithmFPType>(nWS, 16);
        const size_t nBlocks     = nWS / blockSizeWS;
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow = iBlock * blockSizeWS;

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = startRow; i < startRow + blockSizeWS; ++i)
            {
                const size_t wsIndex                       = wsIndices[i];
                const algorithmFPType * const kernelWSData = kernelWS[i];
                yLocal[i]                                  = y[wsIndex];
                gradLocal[i]                               = grad[wsIndex];
                oldAlphaLocal[i]                           = alpha[wsIndex];
                alphaLocal[i]                              = alpha[wsIndex];
                cwLocal[i]                                 = cw[wsIndex];
                kdLocal[i]                                 = kernelWSData[wsIndex % nVectors];
                char Ii                                    = free;
                Ii |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yLocal[i], alphaLocal[i], cwLocal[i]) ? up : free;
                Ii |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yLocal[i], alphaLocal[i], cwLocal[i]) ? low : free;
                Ii |= (yLocal[i] > 0) ? positive : negative;
                I[i] = Ii;

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nWS; ++j)
                {
                    kernelLocal[i * nWS + j] = kernelWSData[wsIndices[j] % nVectors];
                }
            }
        });
    }

    algorithmFPType firstDiff = algorithmFPType(0);
    algorithmFPType delta     = algorithmFPType(0);
    algorithmFPType localEps  = algorithmFPType(0);
    localDiff                 = algorithmFPType(0);
    int Bi                    = -1;
    int Bj                    = -1;

    size_t iter = 0;
    for (; iter < innerMaxIterations; ++iter)
    {
        algorithmFPType GMin  = MaxVal<algorithmFPType>::get();
        algorithmFPType GMax  = -MaxVal<algorithmFPType>::get();
        algorithmFPType GMax2 = -MaxVal<algorithmFPType>::get();

        const algorithmFPType zero(0.0);

        const algorithmFPType * KBiBlock = nullptr;

        if (svmType == SvmType::nu_classification || svmType == SvmType::nu_regression)
        {
            int BiPos = -1;
            int BiNeg = -1;
            int BjPos = -1;
            int BjNeg = -1;

            algorithmFPType GMaxPos  = -MaxVal<algorithmFPType>::get();
            algorithmFPType GMaxNeg  = -MaxVal<algorithmFPType>::get();
            algorithmFPType GMax2Pos = -MaxVal<algorithmFPType>::get();
            algorithmFPType GMax2Neg = -MaxVal<algorithmFPType>::get();

            algorithmFPType deltaPos = algorithmFPType(0);
            algorithmFPType deltaNeg = algorithmFPType(0);

            algorithmFPType GMinPos = HelperTrainSVM<algorithmFPType, cpu>::WSSi(nWS, gradLocal, I, BiPos, SignNuType::positive);
            algorithmFPType GMinNeg = HelperTrainSVM<algorithmFPType, cpu>::WSSi(nWS, gradLocal, I, BiNeg, SignNuType::negative);

            const algorithmFPType KBiBiPos = kdLocal[BiPos];
            const algorithmFPType KBiBiNeg = kdLocal[BiNeg];

            const algorithmFPType * const KBiBlockPos = &kernelLocal[BiPos * nWS];
            const algorithmFPType * const KBiBlockNeg = &kernelLocal[BiNeg * nWS];

            HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(0, nWS, KBiBlockPos, kdLocal, gradLocal, I, GMinPos, KBiBiPos, tau, BjPos, GMaxPos,
                                                            GMax2Pos, deltaPos, SignNuType::positive);
            HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(0, nWS, KBiBlockNeg, kdLocal, gradLocal, I, GMinNeg, KBiBiNeg, tau, BjNeg, GMaxNeg,
                                                            GMax2Neg, deltaNeg, SignNuType::negative);

            if (GMaxPos > GMaxNeg)
            {
                Bi       = BiPos;
                Bj       = BjPos;
                delta    = deltaPos;
                GMin     = GMinPos;
                GMax     = GMaxPos;
                GMax2    = GMax2Pos;
                KBiBlock = KBiBlockPos;
            }
            else
            {
                Bi       = BiNeg;
                Bj       = BjNeg;
                delta    = deltaNeg;
                GMin     = GMinNeg;
                GMax     = GMaxNeg;
                GMax2    = GMax2Neg;
                KBiBlock = KBiBlockNeg;
            }

            localDiff = services::internal::max<cpu, algorithmFPType>(GMax2Pos - GMinPos, GMax2Neg - GMinNeg);
        }
        else
        {
            GMin = HelperTrainSVM<algorithmFPType, cpu>::WSSi(nWS, gradLocal, I, Bi);

            const algorithmFPType KBiBi = kdLocal[Bi];
            KBiBlock                    = &kernelLocal[Bi * nWS];

            HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(0, nWS, KBiBlock, kdLocal, gradLocal, I, GMin, KBiBi, tau, Bj, GMax, GMax2, delta);

            localDiff = GMax2 - GMin;
        }

        if (iter == 0)
        {
            localEps  = services::internal::max<cpu, algorithmFPType>(accuracyThreshold, localDiff * algorithmFPType(1e-1));
            firstDiff = localDiff;
        }
        if (localDiff < localEps)
        {
            break;
        }

        const algorithmFPType yBi  = yLocal[Bi];
        const algorithmFPType yBj  = yLocal[Bj];
        const algorithmFPType cwBi = cwLocal[Bi];
        const algorithmFPType cwBj = cwLocal[Bj];

        /* Update coefficients */
        const algorithmFPType alphaBiDelta = (yBi > zero) ? cwBi - alphaLocal[Bi] : alphaLocal[Bi];
        const algorithmFPType alphaBjDelta =
            services::internal::min<cpu, algorithmFPType>((yBj > zero) ? alphaLocal[Bj] : cwBj - alphaLocal[Bj], delta);
        delta = services::internal::min<cpu, algorithmFPType>(alphaBiDelta, alphaBjDelta);

        /* Update alpha */
        alphaLocal[Bi] += delta * yBi;
        alphaLocal[Bj] -= delta * yBj;

        /* Update up/low sets */
        char IBi = free;
        IBi |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yBi, alphaLocal[Bi], cwBi) ? up : free;
        IBi |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yBi, alphaLocal[Bi], cwBi) ? low : free;
        IBi |= (yBi > 0) ? positive : negative;
        I[Bi] = IBi;

        char IBj = free;
        IBj |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yBj, alphaLocal[Bj], cwBj) ? up : free;
        IBj |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yBj, alphaLocal[Bj], cwBj) ? low : free;
        IBj |= (yBj > 0) ? positive : negative;
        I[Bj] = IBj;

        const algorithmFPType * const KBjBlock = &kernelLocal[Bj * nWS];

        /* Update gradient */
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nWS; i++)
        {
            const algorithmFPType KiBi = KBiBlock[i];
            const algorithmFPType KiBj = KBjBlock[i];
            gradLocal[i] += delta * (KiBi - KiBj);
        }
    }

    localDiff = firstDiff;

    /* Compute diff and scatter to alpha vector */
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nWS; ++i)
    {
        deltaAlpha[i]       = (alphaLocal[i] - oldAlphaLocal[i]) * yLocal[i];
        alpha[wsIndices[i]] = alphaLocal[i];
    }
    return status;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::updateGrad(algorithmFPType ** kernelWS, const algorithmFPType * deltaalpha,
                                                                         algorithmFPType * grad, const size_t nVectors, const size_t nTrainVectors,
                                                                         const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad);

    SafeStatus safeStat;
    const size_t blockSizeGrad = 64;
    const size_t nBlocksGrad   = (nTrainVectors / blockSizeGrad) + !!(nTrainVectors % blockSizeGrad);

    DAAL_INT incX(1);
    DAAL_INT incY(1);

    daal::threader_for(nBlocksGrad, nBlocksGrad, [&](const size_t iBlockGrad) {
        const size_t startRowGrad     = iBlockGrad * blockSizeGrad;
        const size_t nRowsInBlockGrad = (iBlockGrad != nBlocksGrad - 1) ? blockSizeGrad : nTrainVectors - iBlockGrad * blockSizeGrad;
        algorithmFPType * const gradi = &grad[startRowGrad];

        for (size_t i = 0; i < nWS; ++i)
        {
            const algorithmFPType * kernelBlockI = kernelWS[i];
            algorithmFPType deltaalphai          = deltaalpha[i];

            if (startRowGrad < nVectors && startRowGrad + nRowsInBlockGrad > nVectors)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nRowsInBlockGrad; ++j)
                {
                    gradi[j] += deltaalphai * kernelBlockI[(startRowGrad + j) % nVectors];
                }
            }
            else
            {
                const size_t kernelSrartRow = (startRowGrad < nVectors) ? startRowGrad : startRowGrad - nVectors;
                BlasInst<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&nRowsInBlockGrad, &deltaalphai, kernelBlockI + kernelSrartRow, &incX, gradi,
                                                       &incY);
            }
        }
    });

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
bool SVMTrainImpl<thunder, algorithmFPType, cpu>::checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev,
                                                                     const algorithmFPType accuracyThreshold, size_t & sameLocalDiff)
{
    sameLocalDiff =
        internal::MathInst<algorithmFPType, cpu>::sFabs(diff - diffPrev) < accuracyThreshold * accuracyThresholdInner ? sameLocalDiff + 1 : 0;
    if (sameLocalDiff > nNoChanges || diff < accuracyThreshold)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, cpu>::initGrad(const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                                                                       const size_t nVectors, const size_t nTrainVectors, algorithmFPType * const y,
                                                                       algorithmFPType * const alpha, algorithmFPType * grad)
{
    services::Status status;

    TArray<uint32_t, cpu> indicesTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(indicesTArray.get());
    uint32_t * const indices = indicesTArray.get();

    TArray<algorithmFPType, cpu> deltaAlphaTArray(nTrainVectors);
    DAAL_CHECK_MALLOC(deltaAlphaTArray.get());
    algorithmFPType * const deltaAlpha = deltaAlphaTArray.get();

    size_t nNonZeroAlphas = 0;
    for (size_t i = 0; i < nTrainVectors; ++i)
    {
        if (alpha[i] != algorithmFPType(0))
        {
            indices[nNonZeroAlphas]    = static_cast<uint32_t>(i);
            deltaAlpha[nNonZeroAlphas] = alpha[i] * y[i];
            ++nNonZeroAlphas;
        }
    }

    const size_t nBlocks = nNonZeroAlphas / maxBlockSize + !!(nNonZeroAlphas % maxBlockSize);

    auto cachePtr = SVMCache<thunder, lruCache, algorithmFPType, cpu>::create(maxBlockSize, maxBlockSize, nVectors, xTable, kernel, status);

    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
    {
        const size_t startRow     = iBlock * maxBlockSize;
        const size_t nRowsInBlock = (iBlock != nBlocks - 1) ? maxBlockSize : nNonZeroAlphas - iBlock * maxBlockSize;

        if (nRowsInBlock != maxBlockSize)
        {
            status |= cachePtr->resize(nRowsInBlock);
        }

        algorithmFPType ** kernelSOARes = nullptr;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(getRowsBlock);

            status |= cachePtr->getRowsBlock(indices + startRow, nRowsInBlock, kernelSOARes);
        }

        status |= updateGrad(kernelSOARes, deltaAlpha + startRow, grad, nVectors, nTrainVectors, nRowsInBlock);
    }

    cachePtr->clear();

    return status;
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
