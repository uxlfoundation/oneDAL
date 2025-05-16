/* file: em_gmm_dense_default_batch_impl.i */
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

//++
//  Implementation of em algorithm
//--

#include "src/externals/service_math.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_stat.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/em/em_gmm_dense_default_batch_kernel.h"
#include "src/algorithms/em/em_gmm_dense_default_batch_task.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/services/service_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{
/**
 * Function computes expectation-maximization algorithm for Gaussian Mixture Model.
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status EMKernel<algorithmFPType, method, cpu>::compute(NumericTable & dataTable, NumericTable & initialWeights, NumericTable & initialMeans,
                                                                 NumericTable ** initialCovariances, NumericTable & resultWeights,
                                                                 NumericTable & resultMeans, NumericTable ** resultCovariances,
                                                                 NumericTable & resultNIterations, NumericTable & resultGoalFunction,
                                                                 const Parameter & par)
{
    EMKernelTask<algorithmFPType, method, cpu> kernelTask(dataTable, initialWeights, initialMeans, initialCovariances, resultWeights, resultMeans,
                                                          resultCovariances, resultNIterations, resultGoalFunction, par);
    return kernelTask.compute();
};

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status EMKernelTask<algorithmFPType, method, cpu>::compute()
{
    Status s;
    DAAL_CHECK_STATUS(s, initialize())
    DAAL_CHECK_STATUS(s, setStartValues())

    double diff             = 2 * threshold + 1;
    double oldLogLikelyhood = 0;

    daal::tls<Task<algorithmFPType, cpu> *> threadBuffer([=]() -> Task<algorithmFPType, cpu> * {
        return new Task<algorithmFPType, cpu>(dataTable, blockSizeDefault, nFeatures, nComponents, logAlpha, means, covs.get());
    });
    int & iterCounter               = iterCounterArray[0];
    algorithmFPType & logLikelyhood = logLikelyhoodArray[0];
    while (diff > threshold && iterCounter < maxIterations)
    {
        DAAL_CHECK_STATUS(s, covs->computeSigmaInverse(iterCounter))
        algorithmFPType * sqrtInvDetSigma = covs->getLogSqrtInvDetSigma();
        MathInst<algorithmFPType, cpu>::vLog(nComponents, sqrtInvDetSigma, covs->getLogSqrtInvDetSigma());

        MathInst<algorithmFPType, cpu>::vLog(nComponents, alpha, logAlpha); // inplace: same memory as alpha

        logLikelyhood = 0;

        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [=, &threadBuffer, &safeStat](size_t iBlock) {
            size_t j0                     = iBlock * blockSizeDefault;
            size_t nVectorsInCurrentBlock = blockSizeDefault;
            if (iBlock == nBlocks - 1)
            {
                nVectorsInCurrentBlock = nVectors - iBlock * blockSizeDefault;
            }

            Task<algorithmFPType, cpu> * tPtr = threadBuffer.local();
            DAAL_CHECK_THR(tPtr && tPtr->localBuffer, ErrorMemoryAllocationFailed)
            Task<algorithmFPType, cpu> & t = *tPtr;

            Status localStatus = t.next(j0, nVectorsInCurrentBlock);
            DAAL_CHECK_STATUS_THR(localStatus);

            stepE(nVectorsInCurrentBlock, t, par.covarianceStorage);

            t.logLikelyhood += computePartialLogLikelyhood(nVectorsInCurrentBlock, t);

            localStatus |= stepM_partial(nVectorsInCurrentBlock, t, par.covarianceStorage);
            DAAL_CHECK_STATUS_THR(localStatus);
        });
        DAAL_CHECK_SAFE_STATUS()

        setResultToZero();

        threadBuffer.reduce([=, &logLikelyhood](Task<algorithmFPType, cpu> * e) -> void {
            logLikelyhood += e->logLikelyhood;
            e->logLikelyhood = 0;
            for (size_t k = 0; k < nComponents; k++)
            {
                if (e->mergedWSums[k] > MinVal<algorithmFPType>::get())
                {
                    stepM_mergePartialSums(covs->getSigma(k), &e->mergedPartialCP[k * covs->getOneCovSize()], &means[k * nFeatures],
                                           &e->mergedPartialMeans[k * nFeatures], alpha[k], e->mergedWSums[k], nFeatures, covs.get());
                }
            }
            e->setMergedToZero();
        });
        logLikelyhood -= logLikelyhoodCorrection;

        DAAL_CHECK_STATUS(s, stepM_merge(iterCounter))

        if (iterCounter > 0)
        {
            diff = logLikelyhood - oldLogLikelyhood;
        }
        oldLogLikelyhood = logLikelyhood;

        iterCounter++;
    }
    threadBuffer.reduce([=](Task<algorithmFPType, cpu> * v) -> void { delete (v); });
    return s;
}

/* Threshold for vector exp negative args domain  */
template <typename algorithmFPType>
inline algorithmFPType exp_threshold(void)
{
    return algorithmFPType(0.0);
}
template <>
inline float exp_threshold<float>(void)
{
    return float(-75.0);
}
template <>
inline double exp_threshold<double>(void)
{
    return double(-650.0);
}

/**
 * Function computes t.w values that stores weight of each data point belongs to each cluster.
 * t.s is computed by numeric stable log-sum-exp trick.
 */
template <typename algorithmFPType, Method method, CpuType cpu>
void EMKernelTask<algorithmFPType, method, cpu>::stepE(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> & t,
                                                       em_gmm::CovarianceStorageId covType)
{
    const size_t nComponents = t.nComponents;
    const size_t nFeatures   = t.nFeatures;

    if (covType == diagonal)
    {
        for (size_t k = 0; k < nComponents; k++)
        {
            const algorithmFPType * curMean  = &t.means[k * nFeatures];
            const algorithmFPType * invSigma = (t.covs->getSigma())[k];
            const algorithmFPType addition   = t.logAlpha[k] + t.logSqrtInvDetSigma[k];

            for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
            {
                algorithmFPType tp = 0;
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; j++)
                {
                    algorithmFPType x_mu = t.dataBlock[i * nFeatures + j] - curMean[j];
                    tp += x_mu * x_mu * invSigma[j];
                }

                t.p[k * nVectorsInCurrentBlock + i] = addition + -0.5 * tp;
            }
        }
    }
    else
    {
        daal::services::internal::transpose<algorithmFPType, cpu>(t.dataBlock, nVectorsInCurrentBlock, nFeatures, t.trans_data);

        for (size_t k = 0; k < nComponents; k++)
        {
            for (size_t j = 0; j < nFeatures; j++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
                {
                    t.x_mu[j * nVectorsInCurrentBlock + i] =
                        t.trans_data[j * nVectorsInCurrentBlock + i] - t.means[k * nFeatures + j]; // transposed x_mu
                }
            }

            t.covs->multiplyByInverseMatrix(nVectorsInCurrentBlock, k, t.x_mu, t.Ax_mu); // Ax_mu is also transposed

            algorithmFPType addition = t.logAlpha[k] + t.logSqrtInvDetSigma[k];

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
            {
                t.p[k * nVectorsInCurrentBlock + i] = 0.0;
            }

            for (size_t j = 0; j < nFeatures; j++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
                {
                    t.p[k * nVectorsInCurrentBlock + i] += t.x_mu[j * nVectorsInCurrentBlock + i] * t.Ax_mu[j * nVectorsInCurrentBlock + i];
                }
            }

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
            {
                t.p[k * nVectorsInCurrentBlock + i] = addition + -0.5 * t.p[k * nVectorsInCurrentBlock + i];
            }
        }
    }

    t.partLogLikelyhood        = 0;
    algorithmFPType * maxInRow = t.rowSum;
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        maxInRow[i] = t.p[i];
    }

    for (size_t k = 1; k < nComponents; k++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            if (t.p[k * nVectorsInCurrentBlock + i] > maxInRow[i])
            {
                maxInRow[i] = t.p[k * nVectorsInCurrentBlock + i];
            }
        }
    }

    for (size_t k = 0; k < nComponents; k++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.p[k * nVectorsInCurrentBlock + i] -= maxInRow[i];
            if (t.p[k * nVectorsInCurrentBlock + i] < exp_threshold<algorithmFPType>())
            {
                t.p[k * nVectorsInCurrentBlock + i] = exp_threshold<algorithmFPType>();
            }
        }
    }

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        t.partLogLikelyhood += maxInRow[i];
    }

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        maxInRow[i] = 0; // same memory as t.rowSum, set to zero before computing row sum
    }

    MathInst<algorithmFPType, cpu>::vExp(nVectorsInCurrentBlock * nComponents, t.p, t.p);

    for (size_t k = 0; k < nComponents; k++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.rowSum[i] += t.p[k * nVectorsInCurrentBlock + i];
        }
    }

    t.rowSumInv         = t.rowSum;
    algorithmFPType one = 1.0;
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        t.rowSumInv[i] = one / t.rowSum[i];
    }

    for (size_t k = 0; k < nComponents; k++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.p[k * nVectorsInCurrentBlock + i] = t.p[k * nVectorsInCurrentBlock + i] * t.rowSumInv[i];
        }
    }
    t.w = t.p;
}

/**
 * Function computes log likelihood value of for data block. This value contains only logarithm of inversed
 * weighted sum of probabilities of belonging the point to each cluster. To get full log likelihood value,
 * all these values will be summed.
 */

template <typename algorithmFPType, Method method, CpuType cpu>
algorithmFPType EMKernelTask<algorithmFPType, method, cpu>::computePartialLogLikelyhood(const size_t nVectorsInCurrentBlock,
                                                                                        Task<algorithmFPType, cpu> & t)
{
    algorithmFPType * logRowSumInv = t.rowSumInv;
    MathInst<algorithmFPType, cpu>::vLog(nVectorsInCurrentBlock, t.rowSumInv, logRowSumInv);

    algorithmFPType loglikPartial = t.partLogLikelyhood;
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        loglikPartial -= logRowSumInv[i];
    }
    return loglikPartial;
}

/**
 * Function computes from block of data points and weights:
 * 1) sum of weights
 * 2) weighted mean
 * 3) weighted cross product
 * After this computation for current block the results are merged to thread local values
 */
template <typename algorithmFPType, Method method, CpuType cpu>
Status EMKernelTask<algorithmFPType, method, cpu>::stepM_partial(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> & t,
                                                                 em_gmm::CovarianceStorageId covType)
{
    const size_t nFeatures         = t.nFeatures;
    const size_t nElementsOnOneCov = t.covs->getOneCovSize();
    algorithmFPType * dataBlock    = const_cast<algorithmFPType *>(t.trans_data);
    if (covType == diagonal)
    {
        dataBlock = const_cast<algorithmFPType *>(t.dataBlock);
    }

    for (size_t k = 0; k < t.nComponents; k++)
    {
        t.wSums[k] = 0;

        int errcode =
            t.covs->computeThreadPartialResults(dataBlock, &t.w[k * nVectorsInCurrentBlock], t.nFeatures, nVectorsInCurrentBlock, &t.wSums[k],
                                                &t.partialMeans[k * nFeatures], &t.partialCP[k * nElementsOnOneCov], t.w_x_buff);
        if (errcode)
        {
            return Status(Error::create(ErrorEMCovariance, Component, k));
        }

        if (t.wSums[k] > MinVal<algorithmFPType>::get())
        {
            stepM_mergePartialSums(&t.mergedPartialCP[k * nElementsOnOneCov], &t.partialCP[k * nElementsOnOneCov],
                                   &t.mergedPartialMeans[k * nFeatures], &t.partialMeans[k * nFeatures], t.mergedWSums[k], t.wSums[k], nFeatures,
                                   t.covs);
        }
    }
    return Status();
}

/**
 * Function scales merged values of to get result
 */
template <typename algorithmFPType, Method method, CpuType cpu>
Status EMKernelTask<algorithmFPType, method, cpu>::stepM_merge(size_t iteration)
{
    for (size_t k = 0; k < nComponents; k++)
    {
        if (alpha[k] < MinVal<algorithmFPType>::get())
        {
            ErrorPtr e = Error::create(ErrorEMCovariance, Component, k);
            e->addIntDetail(Iteration, iteration + 1);
            return Status(e);
        }
        covs->finalize(k, alpha[k]);
        alpha[k] /= nVectors;
    }
    return Status();
}

/**
 * Sets constants and values
 */
template <typename algorithmFPType, Method method, CpuType cpu>
EMKernelTask<algorithmFPType, method, cpu>::EMKernelTask(NumericTable & dataTable, NumericTable & initialWeights, NumericTable & initialMeans,
                                                         NumericTable ** initialCovariances, NumericTable & resultWeights, NumericTable & resultMeans,
                                                         NumericTable ** resultCovariances, NumericTable & resultNIterations,
                                                         NumericTable & resultGoalFunction, const Parameter & par)
    : dataTable(dataTable),
      initialWeights(initialWeights),
      initialMeans(initialMeans),
      initialCovariances(initialCovariances),
      resultWeights(resultWeights),
      resultMeans(resultMeans),
      resultCovariances(resultCovariances),
      resultNIterations(resultNIterations),
      resultGoalFunction(resultGoalFunction),
      par(par),
      blockSizeDefault(512),
      nFeatures(dataTable.getNumberOfColumns()),
      nVectors(dataTable.getNumberOfRows()),
      threshold(par.accuracyThreshold),
      maxIterations(par.maxIterations),
      nComponents(par.nComponents)
{
    algorithmFPType pi      = 3.1415926535897932384626433;
    logLikelyhoodCorrection = 0.5 * nVectors * nFeatures * MathInst<algorithmFPType, cpu>::sLog(2 * pi);

    nBlocks = nVectors / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != nVectors);
    if (nBlocks == 1)
    {
        blockSizeDefault = nVectors;
    }
    covsPtr.reset(nComponents);
}

template <typename algorithmFPType, Method method, CpuType cpu>
SharedPtr<GmmModel<algorithmFPType, cpu> > EMKernelTask<algorithmFPType, method, cpu>::initializeCovariances()
{
    GmmModelPtr covs;
    if (par.covarianceStorage == diagonal)
    {
        covs = GmmModelPtr(new GmmModelDiagType(nFeatures, nComponents));
    }
    else
    {
        covs = GmmModelPtr(new GmmModelFullType(nFeatures, nComponents));
    }
    covs->setCovRegularizer(par.regularizationFactor);

    WriteRows<algorithmFPType, cpu, NumericTable> * covsNt = covsPtr.get();

    algorithmFPType ** sigma = covs->getSigma();
    for (size_t i = 0; i < nComponents; i++)
    {
        sigma[i] = covsNt[i].set(resultCovariances[i], 0, covs->getNumberOfRowsInCov());
        if (!sigma[i])
        {
            return GmmModelPtr();
        }
    }
    return covs;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status EMKernelTask<algorithmFPType, method, cpu>::initialize()
{
    alpha = weightsBD.set(resultWeights, 0, 1);
    DAAL_CHECK(alpha, ErrorMemoryAllocationFailed);
    logAlpha = alpha;

    means = meansBD.set(resultMeans, 0, nFeatures);
    DAAL_CHECK(means, ErrorMemoryAllocationFailed);

    iterCounterArray = nIterationsBD.set(resultNIterations, 0, 1);
    DAAL_CHECK(iterCounterArray, ErrorMemoryAllocationFailed);
    iterCounterArray[0] = 0;

    logLikelyhoodArray = goalFunctionBD.set(resultGoalFunction, 0, 1);
    DAAL_CHECK(logLikelyhoodArray, ErrorMemoryAllocationFailed);

    covs = initializeCovariances();
    DAAL_CHECK(covs, ErrorMemoryAllocationFailed);

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
void EMKernelTask<algorithmFPType, method, cpu>::setResultToZero()
{
    for (size_t i = 0; i < nComponents; i++)
    {
        alpha[i] = 0;
    }
    for (size_t i = 0; i < nComponents * nFeatures; i++)
    {
        means[i] = 0;
    }
    covs->setToZero();
}

/**
 * Function merges partial cross products, sum of weights and means
 */
template <typename algorithmFPType, Method method, CpuType cpu>
void EMKernelTask<algorithmFPType, method, cpu>::stepM_mergePartialSums(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n,
                                                                        algorithmFPType * mean_m, algorithmFPType & w_n, algorithmFPType & w_m,
                                                                        size_t nFeatures, GmmModel<algorithmFPType, cpu> * covs)
{
    covs->stepM_mergeCovs(cp_n, cp_m, mean_n, mean_m, w_n, w_m, nFeatures);

    algorithmFPType * mom_n = mean_n;
    algorithmFPType * mom_m = mean_m;
    algorithmFPType one_Wnm = 1.0 / (w_n + w_m);

    PRAGMA_VECTOR_UNALIGNED
    for (size_t j = 0; j < nFeatures; j++)
    {
        mom_n[j] = (w_n * mom_n[j] + w_m * mom_m[j]) * one_Wnm;
    }
    w_n += w_m;
}
/**
 * Function merges partial cross products, sum of weights and means
 */
template <typename algorithmFPType, CpuType cpu>
void GmmModelFull<algorithmFPType, cpu>::stepM_mergeCovs(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n,
                                                         algorithmFPType * mean_m, algorithmFPType & w_n, algorithmFPType & w_m, size_t nFeatures)
{
    algorithmFPType one_Wnm = (w_n == (algorithmFPType)0.0) ? 0.0 : (w_n * w_m) / (w_n + w_m);

    for (size_t i = 0; i < nFeatures; i++)
    {
        PRAGMA_VECTOR_UNALIGNED
        for (size_t j = 0; j <= i; j++)
        {
            cp_n[i * nFeatures + j] = cp_n[i * nFeatures + j] + cp_m[i * nFeatures + j]
                                      + one_Wnm * (+mean_n[i] * mean_n[j] + mean_m[i] * mean_m[j] - mean_n[i] * mean_m[j] - mean_m[i] * mean_n[j]);
        }
    }
}

/**
 * Function merges partial cross products, sum of weights and means
 */
template <typename algorithmFPType, CpuType cpu>
void GmmModelDiag<algorithmFPType, cpu>::stepM_mergeCovs(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n,
                                                         algorithmFPType * mean_m, algorithmFPType & w_n, algorithmFPType & w_m, size_t nFeatures)
{
    algorithmFPType one_Wnm = (w_n == (algorithmFPType)0.0) ? 0.0 : (w_n * w_m) / (w_n + w_m);

    for (size_t i = 0; i < nFeatures; i++)
    {
        cp_n[i] = cp_n[i] + cp_m[i] + one_Wnm * (+mean_n[i] * mean_n[i] + mean_m[i] * mean_m[i] - mean_n[i] * mean_m[i] - mean_m[i] * mean_n[i]);
    }
}

/**
 * Function for regularization ill-conditioned covariance matrices. It adds value based on eigen values of matrix to diagonal.
 */
template <typename algorithmFPType, CpuType cpu>
ErrorPtr GmmModelFull<algorithmFPType, cpu>::regularizeCovarianceMatrix(algorithmFPType * cov)
{
    char jobz = 'N';
    char uplo = 'L';

    DAAL_INT lwork  = 2 * nFeatures + 1;
    DAAL_INT liwork = 1;
    DAAL_INT info   = 0;

    TArray<algorithmFPType, cpu> eigenvaluesPtr(nFeatures);
    TArray<algorithmFPType, cpu> diagValuesPtr(nFeatures);
    algorithmFPType * eigenvalues = eigenvaluesPtr.get();
    algorithmFPType * diagValues  = diagValuesPtr.get();

    TArray<algorithmFPType, cpu> workPtr(lwork);
    TArray<DAAL_INT, cpu> iworkPtr(liwork);
    algorithmFPType * work = workPtr.get();
    DAAL_INT * iwork       = iworkPtr.get();
    if (!work || !iwork || !eigenvalues || !diagValues)
    {
        return Error::create(ErrorMemoryAllocationFailed);
    }

    for (size_t i = 0; i < nFeatures; i++)
    {
        diagValues[i] = cov[i * nFeatures + i];
    }

    DAAL_INT p = nFeatures;
    LapackInst<algorithmFPType, cpu>::xxsyevd(&jobz, &uplo, &p, cov, &p, eigenvalues, work, &lwork, iwork, &liwork, &info);
    if (info != 0)
    {
        return Error::create(info < 0 ? ErrorIncorrectInternalFunctionParameter : ErrorEMIllConditionedCovarianceMatrix);
    }

    for (size_t i = 0; i < nFeatures; i++)
    {
        cov[i * nFeatures + i] = diagValues[i];
        for (size_t j = i + 1; j < nFeatures; j++)
        {
            cov[i * nFeatures + j] = cov[j * nFeatures + i];
        }
    }

    if (eigenvalues[0] <= 0.0)
    {
        size_t i = 0;
        while (i < nFeatures && eigenvalues[i] < 0)
        {
            i++;
        }
        if (i == nFeatures)
        {
            return Error::create(ErrorEMNegativeDefinedCovarianceMartix);
        }
    }

    size_t i = 0;
    for (i = 0; i < nFeatures; i++)
    {
        if (eigenvalues[i] > EIGENVALUE_THRESHOLD)
        {
            break;
        }
    }
    if (i == nFeatures)
    {
        return Error::create(ErrorEMIllConditionedCovarianceMatrix);
    }

    //get maximum
    algorithmFPType a              = eigenvalues[i] * covRegularizer;
    algorithmFPType b              = -eigenvalues[0] * (1.0 + covRegularizer);
    algorithmFPType cur_eigenvalue = (a > b) ? a : b;
    for (size_t j = 0; j < nFeatures; j++)
    {
        cov[j * nFeatures + j] += cur_eigenvalue;
    }

    return ErrorPtr();
}

/**
 * Ties to inverse covariance matrices. In case of ill-conditioned matrix try to regularize.
 */
template <typename algorithmFPType, CpuType cpu>
Status GmmModelFull<algorithmFPType, cpu>::computeSigmaInverse(size_t iteration)
{
    typedef LapackInst<algorithmFPType, cpu> lapack;

    algorithmFPType ** invSigma = sigma; //one place for both arrays

    algorithmFPType * sqrtInvDetSigma = logSqrtInvDetSigma;

    daal::tls<algorithmFPType *> sigma_buff(
        [=]() -> algorithmFPType * { return service_scalable_calloc<algorithmFPType, cpu>(nFeatures * nFeatures); });

    SafeStatus safeStat;
    daal::threader_for(nComponents, nComponents, [=, &sigma_buff, &safeStat](size_t iComp) {
        char uplo = 'U';
        DAAL_INT info;
        DAAL_INT lda                = nFeatures;
        DAAL_INT nFeaturesLong      = nFeatures;
        algorithmFPType * pInvSigma = NULL;
        pInvSigma                   = invSigma[iComp];
        algorithmFPType * pSigma    = pInvSigma;

        algorithmFPType * sigmaTmpBuff = sigma_buff.local();
        DAAL_CHECK_THR(sigmaTmpBuff, ErrorMemoryAllocationFailed)
        for (size_t i = 0; i < nFeatures * nFeatures; i++)
        {
            sigmaTmpBuff[i] = pSigma[i];
        }
        lapack::xxpotrf(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
        if (info != 0)
        {
            if (info < 0)
            {
                safeStat.add(Error::create(ErrorIncorrectInternalFunctionParameter, Component, iComp));
                return;
            }
            for (size_t i = 0; i < nFeatures * nFeatures; i++)
            {
                pSigma[i] = sigmaTmpBuff[i];
            }
            ErrorPtr e = regularizeCovarianceMatrix(pSigma);
            if (e)
            {
                e->addIntDetail(Component, iComp);
                safeStat.add(e);
                return;
            }

            lapack::xxpotrf(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
            if (info != 0)
            {
                ErrorPtr e;
                if (info < 0)
                {
                    e = Error::create(ErrorIncorrectInternalFunctionParameter);
                }
                else
                {
                    e = Error::create(ErrorEMIllConditionedCovarianceMatrix);
                }
                e->addIntDetail(Component, iComp);
                e->addIntDetail(Minor, info);
                safeStat.add(e);
                return;
            }
        }
        algorithmFPType sqrtDetSigma = 1;
        for (size_t j = 0; j < nFeatures; j++)
        {
            sqrtDetSigma *= pInvSigma[j * nFeatures + j];
        }
        sqrtDetSigma           = infToBigValue<cpu>(sqrtDetSigma);
        sqrtInvDetSigma[iComp] = 1.0 / sqrtDetSigma;

        lapack::xxpotri(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
        if (info != 0)
        {
            ErrorPtr e;
            if (info < 0)
            {
                e = Error::create(ErrorIncorrectInternalFunctionParameter);
            }
            else
            {
                e = Error::create(ErrorEMMatrixInverse);
            }
            e->addIntDetail(Component, iComp);
            e->addIntDetail(Minor, info);
            safeStat.add(e);
            return;
        }
    });
    sigma_buff.reduce([=](algorithmFPType * v) -> void { service_scalable_free<algorithmFPType, cpu>(v); });
    DAAL_CHECK_SAFE_STATUS()
    return Status();
}

/**
 * Read initial values and copy them to work arrays.
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status EMKernelTask<algorithmFPType, method, cpu>::setStartValues()
{
    ReadRows<algorithmFPType, cpu, NumericTable> bd;
    int result = 0;

    const algorithmFPType * initialWeightsArray = bd.set(initialWeights, 0, 1);
    DAAL_CHECK(initialWeightsArray, ErrorMemoryAllocationFailed);
    if (initialWeightsArray != alpha)
    {
        size_t nCopy = nComponents * sizeof(algorithmFPType);
        result |= daal::services::internal::daal_memcpy_s(alpha, nCopy, initialWeightsArray, nCopy);
    }

    const algorithmFPType * initialMeansArray = bd.set(initialMeans, 0, nComponents);
    DAAL_CHECK(initialMeansArray, ErrorMemoryAllocationFailed);
    if (initialMeansArray != means)
    {
        size_t nCopy = nComponents * nFeatures * sizeof(algorithmFPType);
        result |= daal::services::internal::daal_memcpy_s(means, nCopy, initialMeansArray, nCopy);
    }

    size_t nCopy             = covs->getOneCovSize() * sizeof(algorithmFPType);
    algorithmFPType ** sigma = covs->getSigma();
    for (size_t i = 0; i < nComponents; i++)
    {
        const algorithmFPType * initCov = bd.set(initialCovariances[i], 0, covs->getNumberOfRowsInCov());
        DAAL_CHECK(initCov, ErrorMemoryAllocationFailed);
        if (initCov != sigma[i])
        {
            result |= daal::services::internal::daal_memcpy_s(sigma[i], nCopy, initCov, nCopy);
        }
    }
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal
