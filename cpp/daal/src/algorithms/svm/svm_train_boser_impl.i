/* file: svm_train_boser_impl.i */
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
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_BOSER_IMPL_I__
#define __SVM_TRAIN_BOSER_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_profiler.h"
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
using namespace daal::internal;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<boser, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable,
                                                                    NumericTable & yTable, daal::algorithms::Model * r,
                                                                    const KernelParameter & svmPar)
{
    SVMTrainTask<algorithmFPType, cpu> task(xTable->getNumberOfRows());
    services::Status s = task.setup(svmPar, xTable);
    if (!s) return s;
    DAAL_CHECK_STATUS(s, task.init(svmPar.C, wTable, yTable));
    DAAL_CHECK_STATUS(s, task.compute(svmPar));
    DAAL_CHECK_STATUS(s, task.setResultsToModel(xTable, *static_cast<Model *>(r)));
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::setResultsToModel(const NumericTablePtr & xTable, Model & model) const
{
    SaveResultTask<algorithmFPType, cpu> saveResult(_nVectors, _y.get(), const_cast<algorithmFPType *>(_alpha.get()), _grad.get(),
                                                    SvmType::classification, _cache);
    return saveResult.compute(xTable, model, _cw.get());
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::compute(const KernelParameter & svmPar)
{
    const algorithmFPType eps(svmPar.accuracyThreshold);
    const algorithmFPType tau(svmPar.tau);

    services::Status s;
    size_t nActiveVectors(_nVectors);
    algorithmFPType curEps = MaxVal<algorithmFPType>::get();
    if (!svmPar.doShrinking)
    {
        size_t iter = 0;
        for (; s.ok() && (iter < svmPar.maxIterations) && (eps < curEps); ++iter)
        {
            int Bi, Bj;
            algorithmFPType delta, ma, Ma;
            if (!findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) break;
            s = update(nActiveVectors, Bi, Bj, delta);
        }
        return s;
    }

    bool unshrink = false;
    size_t iter   = 0;
    for (size_t shrinkingIter = 1; (iter < svmPar.maxIterations) && (eps < curEps); ++iter, ++shrinkingIter)
    {
        int Bi, Bj;
        algorithmFPType delta, ma, Ma;
        if (!findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) return s;

        if (curEps < eps)
        {
            /* Check the optimality condition for the task with excluded variables */
            if (unshrink && nActiveVectors < _nVectors) s = reconstructGradient(nActiveVectors);
            if (!s || !findMaximumViolatingPair(nActiveVectors, tau, Bi, Bj, delta, ma, Ma, curEps, s)) return s;
            if (curEps < eps) return s; /* Here if the optimality condition holds for the excluded variables */
            shrinkingIter = 0;
        }
        s = update(nActiveVectors, Bi, Bj, delta);
        if ((shrinkingIter % svmPar.shrinkingStep) == 0)
        {
            if ((!unshrink) && (curEps < 10.0 * eps))
            {
                unshrink = true;
                if (nActiveVectors < _nVectors)
                {
                    s = reconstructGradient(nActiveVectors);
                    if (!s) return s;
                }
            }
            /* Update shrinking flags and do shrinking if needed*/
            if (updateShrinkingFlags(nActiveVectors, ma, Ma) > 0)
            {
                s |= _cache->updateShrinkingRowIndices(nActiveVectors, _flags.get());
                nActiveVectors = doShrink(nActiveVectors);
            }
        }
    }
    if (s) return s;
    if (nActiveVectors < _nVectors) s = reconstructGradient(nActiveVectors);
    return s;
}

/**
 * \brief Working set selection (WSS3) function for the case when the matrix Q is cached.
 *        Select an index j from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[in] tau               parameter of the working set selection algorithm
 * \param[in] Bi                index i from a pair of working set indices B = {i, j}
 * \param[in] GMax              value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha) (see p.1891, eqn.6 in [1])
 * \param[out] Bj               resulting index j
 * \param[out] delta            optimal solution of the sub-problem of size 2:
 *                                  delta =  alpha[i]* - alpha[i]
 *                                  delta = -alpha[j]* + alpha[j]
 *
 * \return The function returns M(alpha) where
 *              M(alpha) = min(-y[i]*grad[i]): i belongs to I_low(alpha)
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::WSSj(size_t nActiveVectors, algorithmFPType tau, int Bi, algorithmFPType GMin, int & Bj,
                                                          algorithmFPType & delta, algorithmFPType & res) const
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair.WSSj);

    Bj                    = -1;
    algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
    algorithmFPType GMax  = -fpMax; // some big negative number
    algorithmFPType GMax2 = -fpMax; // some big negative number

    algorithmFPType Kii = _kernelDiag[Bi];

    services::Status s;
    size_t nBlocks = nActiveVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < nActiveVectors)
    {
        nBlocks++;
    }
    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        size_t jStart = iBlock * kernelFunctionBlockSize;
        size_t jEnd   = jStart + kernelFunctionBlockSize;
        if (jEnd > nActiveVectors)
        {
            jEnd = nActiveVectors;
        }

        const algorithmFPType * KiBlock = nullptr;
        s                               = _cache->getRowBlock(Bi, jStart, (jEnd - jStart), KiBlock);
        if (!s) break;

        int Bj_local = -1;
        algorithmFPType GMax_local, GMax2_local, delta_local;
        HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(jStart, jEnd, KiBlock, _kernelDiag.get(), _grad.get(), _flags.get(), GMin, Kii, tau, Bj_local,
                                                        GMax_local, GMax2_local, delta_local);

        if (GMax_local > GMax)
        {
            GMax  = GMax_local;
            Bj    = Bj_local;
            delta = delta_local;
        }
        if (GMax2_local > GMax2)
        {
            GMax2 = GMax2_local;
        }
    }
    res = GMax2;
    return s;
}

template <typename algorithmFPType, CpuType cpu>
bool SVMTrainTask<algorithmFPType, cpu>::findMaximumViolatingPair(size_t nActiveVectors, algorithmFPType tau, int & Bi, int & Bj,
                                                                  algorithmFPType & delta, algorithmFPType & ma, algorithmFPType & Ma,
                                                                  algorithmFPType & curEps, services::Status & s) const
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair);

    Bi = -1;
    ma = HelperTrainSVM<algorithmFPType, cpu>::WSSi(nActiveVectors, _grad.get(), _flags.get(), Bi);
    if (Bi == -1) return false;

    Bj = -1;
    s |= WSSj(nActiveVectors, tau, Bi, ma, Bj, delta, Ma);
    curEps = Ma - ma;
    return (Bj != -1);
}

template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::update(size_t nActiveVectors, int Bi, int Bj, algorithmFPType delta)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad);

    /* Update alpha */
    algorithmFPType newDeltai, newDeltaj;
    updateAlpha(Bi, Bj, delta, newDeltai, newDeltaj);
    updateFlag(Bj);
    updateFlag(Bi);

    const algorithmFPType * y = _y.get();
    const algorithmFPType dyi = newDeltai * y[Bi];
    const algorithmFPType dyj = newDeltaj * y[Bj];

    /* Update gradient */
    const size_t blockSize = (kernelFunctionBlockSize >> 1); // 2 rows from kernel function matrix are used
    size_t nBlocks         = nActiveVectors / blockSize;
    if (nBlocks * blockSize < nActiveVectors) ++nBlocks;

    algorithmFPType * grad = _grad.get();
    services::Status s;

    for (size_t iBlock = 0; s.ok() && (iBlock < nBlocks); iBlock++)
    {
        size_t tStart = iBlock * blockSize;
        size_t tEnd   = tStart + blockSize;
        if (tEnd > nActiveVectors)
        {
            tEnd = nActiveVectors;
        }

        const algorithmFPType * KiBlock = nullptr;
        const algorithmFPType * KjBlock = nullptr;
        s                               = _cache->getTwoRowsBlock(Bi, Bj, tStart, (tEnd - tStart), KiBlock, KjBlock);

        for (size_t t = tStart; t < tEnd; t++)
        {
            grad[t] += dyi * KiBlock[t - tStart];
            grad[t] += dyj * KjBlock[t - tStart];
        }
    }
    return s;
} // namespace internal

/**
 * \brief Update the array of classification coefficients. Step 4 of the Algorithm 1 in [1].
 *
 * \param[in] C          Upper bound in constraints of the quadratic optimization problem
 * \param[in] Bi         Index i from a pair of working set indices B = {i, j}
 * \param[in] Bj         Index j from a pair of working set indices B = {i, j}
 * \param[in] delta      Estimated difference between old and new value of the classification coefficients
 * \param[out] newDeltai Resulting difference between old and new value of the Bi-th classification coefficient
 * \param[out] newDeltaj Resulting difference between old and new value of the Bj-th classification coefficient
 */
template <typename algorithmFPType, CpuType cpu>
inline void SVMTrainTask<algorithmFPType, cpu>::updateAlpha(int Bi, int Bj, algorithmFPType delta, algorithmFPType & newDeltai,
                                                            algorithmFPType & newDeltaj)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateAlpha);
    const algorithmFPType * cw = _cw.get();

    const algorithmFPType oldAlphai = _alpha[Bi];
    const algorithmFPType oldAlphaj = _alpha[Bj];
    const algorithmFPType yi        = _y[Bi];
    const algorithmFPType yj        = _y[Bj];
    const algorithmFPType cwi       = _cw[Bi];
    const algorithmFPType cwj       = _cw[Bj];

    const algorithmFPType alphaBiDelta = (yi > 0.0f) ? cwi - oldAlphai : oldAlphai;
    const algorithmFPType alphaBjDelta = services::internal::min<cpu, algorithmFPType>((yj > 0.0f) ? oldAlphaj : cwj - oldAlphaj, delta);
    delta                              = services::internal::min<cpu, algorithmFPType>(alphaBiDelta, alphaBjDelta);

    algorithmFPType newAlphai = oldAlphai + yi * delta;
    algorithmFPType newAlphaj = oldAlphaj - yj * delta;

    _alpha[Bj] = newAlphaj;
    _alpha[Bi] = newAlphai;
    newDeltai  = newAlphai - oldAlphai;
    newDeltaj  = newAlphaj - oldAlphaj;
}

/**
 * \brief Performs shrinking by moving the variables that corresponf to the shrunk feature vectors
 *        at the end of the respective arrays.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \return Number of observations that remain active (not shrunk) after shrinking
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainTask<algorithmFPType, cpu>::doShrink(size_t nActiveVectors)
{
    size_t i                     = 0;
    size_t j                     = nActiveVectors - 1;
    algorithmFPType * y          = _y.get();
    algorithmFPType * grad       = _grad.get();
    algorithmFPType * alpha      = _alpha.get();
    algorithmFPType * kernelDiag = _kernelDiag.get();
    char * flags                 = _flags.get();
    while (i < j)
    {
        while (!(flags[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((flags[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, char>(flags[i], flags[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(y[i], y[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(alpha[i], alpha[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(grad[i], grad[j]);
        daal::services::internal::swap<cpu, algorithmFPType>(kernelDiag[i], kernelDiag[j]);
    }
    return i;
}

/**
 * \brief Update the array of flags that specify if the feature vector will be shrunk or not.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] C              Upper bound in constraints of the quadratic optimization problem
 * \param[in] ma             Value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)  (see p.1891, eqn.6 in [1])
 * \param[in] Ma             Value of M(alpha) = min(-y[i]*grad[i]): i belongs to I_LOW (alpha)
 * \return Number of observations to be shrunk
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainTask<algorithmFPType, cpu>::updateShrinkingFlags(size_t nActiveVectors, algorithmFPType ma, algorithmFPType Ma)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateShrinkingFlags);

    const algorithmFPType * const y     = _y.get();
    const algorithmFPType * const grad  = _grad.get();
    const algorithmFPType * const alpha = _alpha.get();
    const algorithmFPType * const cw    = _cw.get();

    char * flags   = _flags.get();
    size_t nShrink = 0;
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        flags[i] &= (~shrink);
        const algorithmFPType gradi = grad[i];
        if (gradi < ma)
        {
            if (alpha[i] <= 0.0 && y[i] == -1.0)
            {
                flags[i] |= shrink;
                nShrink++;
            }
            if (alpha[i] >= cw[i] && y[i] == 1.0)
            {
                flags[i] |= shrink;
                nShrink++;
            }
        }
        if (gradi > Ma)
        {
            if (alpha[i] <= 0.0 && y[i] == 1.0)
            {
                flags[i] |= shrink;
                nShrink++;
            }
            if (alpha[i] >= cw[i] && y[i] == -1.0)
            {
                flags[i] |= shrink;
                nShrink++;
            }
        }
    }
    return nShrink;
}

/**
 * \brief Recompute the values of optimization function gradient for the vectors that were shrunk
 *
 * \param[in] nVectors       Number of observations in a training data set
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::reconstructGradient(size_t & nActiveVectors)
{
    size_t nBlocks = _nVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < _nVectors)
    {
        nBlocks++;
    }
    algorithmFPType * grad        = _grad.get();
    const algorithmFPType * y     = _y.get();
    const algorithmFPType * alpha = _alpha.get();
    services::Status s;
    for (size_t i = nActiveVectors; i < _nVectors; i++)
    {
        grad[i] = -y[i];

        for (size_t jBlock = 0; s.ok() && (jBlock < nBlocks); jBlock++)
        {
            size_t jStart = jBlock * kernelFunctionBlockSize;
            size_t jEnd   = jStart + kernelFunctionBlockSize;
            if (jEnd > _nVectors)
            {
                jEnd = _nVectors;
            }

            const algorithmFPType * cacheRow = nullptr;
            s                                = _cache->getRowBlock(i, jStart, (jEnd - jStart), cacheRow);
            for (size_t j = jStart; j < jEnd; j++)
            {
                grad[i] += y[j] * cacheRow[j - jStart] * alpha[j];
            }
        }
    }
    nActiveVectors = _nVectors;
    return s;
}

/**
 * \brief Construct the structure that stores the intermediate data used in SVM training
 *
 * \param[in] svmPar        Parameters of the algorithm
 * \param[in] xTable        Pointer to numeric table that contains input data set
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::setup(const KernelParameter & svmPar, const NumericTablePtr & xTable)
{
    _alpha.reset(_nVectors);
    daal::services::internal::service_memset<algorithmFPType, cpu>(_alpha.get(), algorithmFPType(0.0), _nVectors);
    _flags.reset(_nVectors);
    daal::services::internal::service_memset<char, cpu>(_flags.get(), char(0), _nVectors);
    _y.reset(_nVectors);
    _grad.reset(_nVectors);
    _cw.reset(_nVectors);
    _kernelDiag.reset(_nVectors);
    DAAL_CHECK_MALLOC(_alpha.get() && _flags.get() && _y.get() && _grad.get() && _cw.get() && _kernelDiag.get());

    kernel_function::KernelIfacePtr kernel = svmPar.kernel->clone();
    size_t cacheSize                       = svmPar.cacheSize;
    services::Status s;
    if (cacheSize >= _nVectors * _nVectors * sizeof(algorithmFPType))
    {
        _cache = SVMCache<boser, simpleCache, algorithmFPType, cpu>::create(_nVectors, svmPar.doShrinking, xTable, kernel, s);
    }
    else
    {
        cacheSize = kernelFunctionBlockSize;
        _cache    = SVMCache<boser, noCache, algorithmFPType, cpu>::create(cacheSize, _nVectors, svmPar.doShrinking, xTable, kernel, s);
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
SVMTrainTask<algorithmFPType, cpu>::~SVMTrainTask()
{
    delete _cache;
}

/**
 * \brief Initialize the intermediate data used in SVM training
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
*  \param[in] wTable        Pointer to numeric table that contains weights
*  \param[in] yTable        Pointer to numeric table that contains class labels
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainTask<algorithmFPType, cpu>::init(algorithmFPType C, const NumericTablePtr & wTable, NumericTable & yTable)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(init);

    ReadColumns<algorithmFPType, cpu> mtW(wTable.get(), 0, 0, _nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtW);
    const algorithmFPType * const weights = mtW.get();

    ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, 0, _nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtY);
    const algorithmFPType * const yIn = mtY.get();

    algorithmFPType * const grad = _grad.get();
    algorithmFPType * const y    = _y.get();
    algorithmFPType * const cw   = _cw.get();
    for (size_t i = 0; i < _nVectors; i++)
    {
        y[i]    = yIn[i] == 0 ? algorithmFPType(-1) : yIn[i];
        grad[i] = -y[i];
        cw[i]   = weights ? weights[i] * C : C;
        updateFlag(i);
    }

    services::Status s;
    algorithmFPType * kernelDiag = _kernelDiag.get();
    for (size_t i = 0; s.ok() && (i < _nVectors); i++)
    {
        const algorithmFPType * KiiPtr = nullptr;
        s                              = _cache->getRowBlock(i, i, 1, KiiPtr);
        kernelDiag[i]                  = *KiiPtr;
    }
    return s;
}

/**
 * \brief Update the flag that specify the status of the feature vector in the input data set
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
 * \param[in] index Index of the feature vector
 */
template <typename algorithmFPType, CpuType cpu>
inline void SVMTrainTask<algorithmFPType, cpu>::updateFlag(size_t index)
{
    char Ii                      = _flags[index];
    const algorithmFPType alphai = _alpha[index];
    const algorithmFPType yi     = _y[index];
    const algorithmFPType cwi    = _cw[index];
    Ii &= (char)shrink;

    Ii |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yi, alphai, cwi) ? up : free;
    Ii |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yi, alphai, cwi) ? low : free;
    Ii |= (yi > 0) ? positive : negative;
    _flags[index] = Ii;
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] flags          Array of flags that describe the status of feature vectors
 * \return                   services::Status of the call
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMCache<boser, noCache, algorithmFPType, cpu>::updateShrinkingRowIndices(size_t nActiveVectors, const char * flags)
{
    size_t i = 0;
    size_t j = nActiveVectors - 1;
    while (i < j)
    {
        while (!(flags[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((flags[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, size_t>(_shrinkingRowIndices[i], _shrinkingRowIndices[j]);
        i++;
        j--;
    }
    return services::Status();
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array and
 *        re-order rows and columns in the cache accordingly
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] flags          Array of flags that describe the status of feature vectors
 * \return                   services::Status of the call
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SVMCache<boser, simpleCache, algorithmFPType, cpu>::updateShrinkingRowIndices(size_t nActiveVectors, const char * flags)
{
    size_t i   = 0;
    size_t j   = nActiveVectors - 1;
    int result = 0;
    while (i < j)
    {
        while (!(flags[i] & shrink) && i < nActiveVectors - 1) i++;
        while ((flags[j] & shrink) && j > 0) j--;
        if (i >= j) break;
        daal::services::internal::swap<cpu, size_t>(_shrinkingRowIndices[i], _shrinkingRowIndices[j]);

        algorithmFPType cacheii = _cache[i * _lineSize + i];
        algorithmFPType cacheij = _cache[i * _lineSize + j];
        algorithmFPType cachejj = _cache[j * _lineSize + j];

        /* Swap i-th and j-th row in cache */
        size_t lineSizeInBytes = _lineSize * sizeof(algorithmFPType);
        result |= daal::services::internal::daal_memcpy_s(_tmp.get(), lineSizeInBytes, _cache.get() + i * _lineSize, lineSizeInBytes);
        result |=
            daal::services::internal::daal_memcpy_s(_cache.get() + i * _lineSize, lineSizeInBytes, _cache.get() + j * _lineSize, lineSizeInBytes);
        result |= daal::services::internal::daal_memcpy_s(_cache.get() + j * _lineSize, lineSizeInBytes, _tmp.get(), lineSizeInBytes);

        /* Swap i-th and j-th column in cache */
        for (size_t k = 0; k < _nLines; k++)
        {
            daal::services::internal::swap<cpu, algorithmFPType>(_cache[i + k * _lineSize], _cache[j + k * _lineSize]);
        }

        // maybe not needed
        _cache[i * _lineSize + i] = cachejj;
        _cache[j * _lineSize + j] = cacheii;
        _cache[j * _lineSize + i] = cacheij;
        _cache[i * _lineSize + j] = cacheij;
        i++;
        j--;
    }
    return (!result) ? services::Status() : services::Status(ErrorMemoryCopyFailedInternal);
}
} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
