/* file: em_gmm_dense_default_batch_task.h */
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
//  Implementation of task for the em algorithm
//--

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_TASK_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_TASK_H__

#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_stat.h"
#include "src/externals/service_math.h"
#include "src/algorithms/service_sort.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class GmmModel
{
public:
    DAAL_NEW_DELETE();
    GmmModel(size_t _nFeatures, size_t _nComponents)
        : nFeatures(_nFeatures), nComponents(_nComponents), sigma(nullptr), sigmaPtr(_nComponents), logSqrtInvDetSigmaPtr(_nComponents)
    {
        sigma                = sigmaPtr.get();
        logSqrtInvDetSigma   = logSqrtInvDetSigmaPtr.get();
        EIGENVALUE_THRESHOLD = 1000 * MinVal<algorithmFPType>::get();
    }
    virtual ~GmmModel() {}
    algorithmFPType ** getSigma() { return sigma; }
    algorithmFPType * getSigma(size_t index)
    {
        if (sigma)
        {
            return sigma[index];
        }
        else
        {
            return nullptr;
        }
    }
    algorithmFPType * getLogSqrtInvDetSigma() { return logSqrtInvDetSigma; }

    void setToZero()
    {
        size_t nElements = getOneCovSize();
        for (size_t i = 0; i < nComponents; i++)
        {
            for (size_t j = 0; j < nElements; j++)
            {
                sigma[i][j] = 0;
            }
        }
    }

    virtual size_t getOneCovSize()                                                                                             = 0;
    virtual size_t getNumberOfRowsInCov()                                                                                      = 0;
    virtual void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType * X, algorithmFPType * CovX) = 0;
    virtual Status computeSigmaInverse(size_t iteration)                                                                       = 0;
    virtual int computeThreadPartialResults(algorithmFPType * data, algorithmFPType * weights, size_t nFeatures, size_t nElements,
                                            algorithmFPType * sumOfWeights, algorithmFPType * partialMean, algorithmFPType * partialCovs,
                                            algorithmFPType * w_x_buf)                                                         = 0;
    virtual void stepM_mergeCovs(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n, algorithmFPType * mean_m,
                                 algorithmFPType & w_n, algorithmFPType & w_m, size_t nFeatures)                               = 0;
    virtual void finalize(size_t k, algorithmFPType denominator)                                                               = 0;
    virtual void setCovRegularizer(double _covRegularizer) { covRegularizer = _covRegularizer; }

protected:
    algorithmFPType ** sigma;
    algorithmFPType * logSqrtInvDetSigma;
    size_t nComponents;
    size_t nFeatures;
    TArray<algorithmFPType *, cpu> sigmaPtr;
    TArray<algorithmFPType, cpu> logSqrtInvDetSigmaPtr;
    double covRegularizer;
    algorithmFPType EIGENVALUE_THRESHOLD;
};

template <typename algorithmFPType, CpuType cpu>
class GmmModelFull : public GmmModel<algorithmFPType, cpu>
{
public:
    using GmmModel<algorithmFPType, cpu>::nFeatures;
    using GmmModel<algorithmFPType, cpu>::nComponents;
    using GmmModel<algorithmFPType, cpu>::sigma;
    using GmmModel<algorithmFPType, cpu>::logSqrtInvDetSigma;
    using GmmModel<algorithmFPType, cpu>::covRegularizer;
    using GmmModel<algorithmFPType, cpu>::EIGENVALUE_THRESHOLD;
    typedef BlasInst<algorithmFPType, cpu> blas;

    GmmModelFull(size_t _nFeatures, size_t _nComponents) : GmmModel<algorithmFPType, cpu>(_nFeatures, _nComponents) {}
    size_t getOneCovSize() { return nFeatures * nFeatures; }
    size_t getNumberOfRowsInCov() { return nFeatures; }
    Status computeSigmaInverse(size_t iteration);
    void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType * X, algorithmFPType * CovX)
    {
        char side                  = 'R';
        char uplo                  = 'U';
        DAAL_INT m                 = nVectorsInCurrentBlock;
        DAAL_INT n                 = nFeatures;
        algorithmFPType alphaCoeff = 1.0;
        algorithmFPType betaCoeff  = 0.0;

        algorithmFPType * invSigma = sigma[k];
        blas::xxsymm(&side, &uplo, &m, &n, &alphaCoeff, invSigma, &n, X, &m, &betaCoeff, CovX, &m);
    }

    int computeThreadPartialResults(algorithmFPType * data, algorithmFPType * weights, size_t nFeatures, size_t nElements,
                                    algorithmFPType * sumOfWeights, algorithmFPType * partialMean, algorithmFPType * partialCovs,
                                    algorithmFPType * w_x_buf)
    {
        StatisticsInst<algorithmFPType, cpu>::xxcp_weight_byrows(weights, data, nElements, nFeatures, w_x_buf, *sumOfWeights, partialMean,
                                                                 partialCovs);
        return 0;
    }

    void finalize(size_t k, algorithmFPType denominator)
    {
        algorithmFPType multiplier = 1.0 / denominator;
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                sigma[k][i * nFeatures + j] *= multiplier;
                sigma[k][j * nFeatures + i] = sigma[k][i * nFeatures + j];
            }
            sigma[k][i * nFeatures + i] *= multiplier;
        }
    }

    void stepM_mergeCovs(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n, algorithmFPType * mean_m, algorithmFPType & w_n,
                         algorithmFPType & w_m, size_t nFeatures);

    ErrorPtr regularizeCovarianceMatrix(algorithmFPType * cov);
};

template <typename algorithmFPType, CpuType cpu>
class GmmModelDiag : public GmmModel<algorithmFPType, cpu>
{
public:
    using GmmModel<algorithmFPType, cpu>::nFeatures;
    using GmmModel<algorithmFPType, cpu>::nComponents;
    using GmmModel<algorithmFPType, cpu>::sigma;
    using GmmModel<algorithmFPType, cpu>::logSqrtInvDetSigma;
    using GmmModel<algorithmFPType, cpu>::covRegularizer;
    using GmmModel<algorithmFPType, cpu>::EIGENVALUE_THRESHOLD;
    GmmModelDiag(size_t _nFeatures, size_t _nComponents) : GmmModel<algorithmFPType, cpu>(_nFeatures, _nComponents) {}
    size_t getOneCovSize() { return nFeatures; }
    size_t getNumberOfRowsInCov() { return 1; }
    void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType * X, algorithmFPType * CovX)
    {
        algorithmFPType * invSigma = sigma[k];
        for (size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            for (size_t j = 0; j < nFeatures; j++)
            {
                CovX[i * nFeatures + j] = X[i * nFeatures + j] * invSigma[j];
            }
        }
    }
    ErrorPtr regularizeCovarianceMatrix(algorithmFPType * cov)
    {
        TArray<algorithmFPType, cpu> sortedCovsPtr(nFeatures);
        algorithmFPType * sortedCovs = sortedCovsPtr.get();
        if (!sortedCovs)
        {
            return ErrorPtr(new Error(ErrorMemoryAllocationFailed));
        }
        for (size_t j = 0; j < nFeatures; j++)
        {
            sortedCovs[j] = cov[j];
        }
        daal::algorithms::internal::qSort<algorithmFPType, cpu>(nFeatures, sortedCovs);

        algorithmFPType * eigenvalues = sortedCovs;
        size_t i                      = 0;
        for (i = 0; i < nFeatures; i++)
        {
            if (eigenvalues[i] > EIGENVALUE_THRESHOLD)
            {
                break;
            }
        }
        if (i == nFeatures)
        {
            return ErrorPtr(new Error(ErrorEMIllConditionedCovarianceMatrix));
        }

        //get maximum
        algorithmFPType a              = eigenvalues[i] * covRegularizer;
        algorithmFPType b              = -eigenvalues[0] * (1.0 + covRegularizer);
        algorithmFPType cur_eigenvalue = (a > b) ? a : b;
        for (size_t j = 0; j < nFeatures; j++)
        {
            cov[j] += cur_eigenvalue;
        }
        return ErrorPtr();
    }

    Status computeSigmaInverse(size_t iteration)
    {
        algorithmFPType ** invSigma       = sigma;
        algorithmFPType * sqrtInvDetSigma = logSqrtInvDetSigma;

        for (size_t k = 0; k < nComponents; k++)
        {
            for (size_t j = 0; j < nFeatures; j++)
            {
                if (sigma[k][j] < EIGENVALUE_THRESHOLD)
                {
                    ErrorPtr e = regularizeCovarianceMatrix(sigma[k]);
                    if (e)
                    {
                        e->addIntDetail(Iteration, iteration + 1);
                        return Status(e);
                    }
                }
            }
            algorithmFPType determ = 1;
            for (size_t j = 0; j < nFeatures; j++)
            {
                determ *= sigma[k][j];
                invSigma[k][j] = 1.0 / sigma[k][j];
            }
            determ             = infToBigValue<cpu>(determ);
            sqrtInvDetSigma[k] = 1.0 / MathInst<algorithmFPType, cpu>::sSqrt(determ);
        }
        return Status();
    }

    int computeThreadPartialResults(algorithmFPType * data, algorithmFPType * weights, size_t nFeatures, size_t nElements,
                                    algorithmFPType * sumOfWeights, algorithmFPType * partialMean, algorithmFPType * partialCovs,
                                    algorithmFPType * w_x_buf)
    {
        __int64 matCompMethod = __DAAL_VSL_SS_METHOD_FAST;
        int errcode = StatisticsInst<algorithmFPType, cpu>::xxvar_weight(data, nFeatures, nElements, weights, sumOfWeights, partialMean, partialCovs,
                                                                         matCompMethod);
        return errcode;
    }

    void finalize(size_t k, algorithmFPType denominator)
    {
        algorithmFPType multiplier = 1.0 / denominator;
        for (size_t i = 0; i < nFeatures; i++)
        {
            sigma[k][i] *= multiplier;
        }
    }

    void stepM_mergeCovs(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n, algorithmFPType * mean_m, algorithmFPType & w_n,
                         algorithmFPType & w_m, size_t nFeatures);
};

template <typename algorithmFPType, CpuType cpu>
struct Task
{
    DAAL_NEW_DELETE()

    Task(NumericTable & _dataTable, size_t blockSizeDefault, size_t _nFeatures, size_t _nComponents,
         algorithmFPType * _logAlpha, //placed in alpha memory
         algorithmFPType * _means, GmmModel<algorithmFPType, cpu> * _covs)
        : dataTable(&_dataTable),
          dataBlock(nullptr),
          logAlpha(_logAlpha),
          means(_means),
          covs(_covs),
          invSigma(_covs->getSigma()),
          logSqrtInvDetSigma(_covs->getLogSqrtInvDetSigma()),
          nFeatures(_nFeatures),
          nComponents(_nComponents),
          logLikelyhood(0)
    {
        size_t sizeOfOneCov           = covs->getOneCovSize();
        size_t memorySizeForOneThread = blockSizeDefault * nFeatures +   /* x_mu   */
                                        blockSizeDefault * nFeatures +   /* Ax_mu  */
                                        blockSizeDefault * nComponents + /* p      */
                                        blockSizeDefault +               /* rowSum */
                                        nComponents +                    /* wSums */
                                        nComponents * nFeatures +        /* partialMeans */
                                        nComponents * sizeOfOneCov +     /* partialCP */
                                        nComponents +                    /* mergedWSums */
                                        nComponents * nFeatures +        /* mergedPartialMeans */
                                        nComponents * sizeOfOneCov +     /* mergedPartialCP */
                                        blockSizeDefault * nFeatures +   /* trans_data */
                                        blockSizeDefault * nFeatures;    /* w_x buff */

        threadBufferPtr.reset(memorySizeForOneThread);
        localBuffer = threadBufferPtr.get();
        if (!localBuffer)
        {
            return;
        }

        x_mu         = localBuffer;
        Ax_mu        = &x_mu[blockSizeDefault * nFeatures];
        p            = &Ax_mu[blockSizeDefault * nFeatures];
        rowSum       = &p[blockSizeDefault * nComponents];
        wSums        = &rowSum[blockSizeDefault];
        partialMeans = &wSums[nComponents];
        partialCP    = &partialMeans[nComponents * nFeatures];

        mergedWSums        = &partialCP[nComponents * sizeOfOneCov];
        mergedPartialMeans = &mergedWSums[nComponents];
        mergedPartialCP    = &mergedPartialMeans[nComponents * nFeatures];
        trans_data         = &mergedPartialCP[nComponents * sizeOfOneCov];
        w_x_buff           = &trans_data[blockSizeDefault * nFeatures];
        setMergedToZero();
    }

    void setMergedToZero()
    {
        size_t sizeOfOneCov = covs->getOneCovSize();
        for (size_t i = 0; i < nComponents; i++)
        {
            mergedWSums[i] = 0;
        }
        for (size_t i = 0; i < nComponents * nFeatures; i++)
        {
            mergedPartialMeans[i] = 0;
        }
        for (size_t i = 0; i < nComponents * sizeOfOneCov; i++)
        {
            mergedPartialCP[i] = 0;
        }
    }

    Status next(size_t j0, size_t nVectorsInCurrentBlock)
    {
        dataTableBD.set(dataTable, j0, nVectorsInCurrentBlock);
        DAAL_CHECK_BLOCK_STATUS(dataTableBD)
        dataBlock = dataTableBD.get();
        return Status();
    }

    NumericTable * dataTable;
    algorithmFPType * localBuffer;
    const algorithmFPType * dataBlock;
    ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD;
    TArray<algorithmFPType, cpu> threadBufferPtr;
    algorithmFPType logLikelyhood;

    algorithmFPType * x_mu;
    algorithmFPType * Ax_mu;
    algorithmFPType * w;
    algorithmFPType * p;
    algorithmFPType * rowSum;
    algorithmFPType * rowSumInv;

    algorithmFPType * logAlpha;
    algorithmFPType * means;
    algorithmFPType ** invSigma;
    algorithmFPType * logSqrtInvDetSigma;
    algorithmFPType partLogLikelyhood;

    algorithmFPType * wSums;
    algorithmFPType * partialMeans;
    algorithmFPType * partialCP;

    algorithmFPType * mergedWSums;
    algorithmFPType * mergedPartialMeans;
    algorithmFPType * mergedPartialCP;

    algorithmFPType * trans_data;
    algorithmFPType * w_x_buff;

    GmmModel<algorithmFPType, cpu> * covs;

    size_t nFeatures;
    size_t nComponents;
};

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
