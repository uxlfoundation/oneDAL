/* file: naivebayes_train_impl.i */
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
//  Implementation of Naive Bayes algorithm.
//
//  Based on paper: Tackling the Poor Assumptions of Naive Bayes Text Classifiers
//  Url: http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
//--
*/

#ifndef __NAIVEBAYES_TRAIN_FAST_I__
#define __NAIVEBAYES_TRAIN_FAST_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "data_management/data/csr_numeric_table.h"

#include "src/services/service_data_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#if (__CPUID__(DAAL_CPU) >= __avx512__)

    #define _CALLOC_ service_scalable_calloc
    #define _FREE_   service_scalable_free

#else

    #define _CALLOC_ service_calloc
    #define _FREE_   service_free

#endif

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
struct localDataCollector
{};

template <typename algorithmFPType, CpuType cpu>
struct localDataCollector<algorithmFPType, defaultDense, cpu>
{
    size_t _p;
    size_t _c;

    ReadRows<algorithmFPType, cpu> rrData;
    ReadRows<int, cpu> rrClass;

    algorithmFPType * n_ci;

    localDataCollector(size_t p, size_t c, NumericTable * _ntData, NumericTable * _ntClass, algorithmFPType * local_n_ci)
        : _p(p), _c(c), rrData(_ntData), rrClass(_ntClass), n_ci(local_n_ci)
    {}

    size_t getBlockSize(size_t n) { return 256; }

    Status addData(size_t nStart, size_t blockSize)
    {
        rrData.next(nStart, blockSize);
        DAAL_CHECK_BLOCK_STATUS(rrData);
        rrClass.next(nStart, blockSize);
        DAAL_CHECK_BLOCK_STATUS(rrClass);

        const algorithmFPType * data = rrData.get();
        const int * predefClass      = rrClass.get();

        for (size_t j = 0; j < blockSize; j++)
        {
            int cl = predefClass[j];
            DAAL_ASSERT(cl < _c);
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < _p; i++)
            {
                n_ci[cl * _p + i] += data[j * _p + i];
            }
        }

        return Status();
    }
};

template <typename algorithmFPType, CpuType cpu>
struct localDataCollector<algorithmFPType, fastCSR, cpu>
{
    size_t _p;
    size_t _c;

    ReadRowsCSR<algorithmFPType, cpu> rrData;
    ReadRows<int, cpu> rrClass;

    algorithmFPType * n_ci;

    localDataCollector(size_t p, size_t c, NumericTable * _ntData, NumericTable * _ntClass, algorithmFPType * local_n_ci)
        : _p(p), _c(c), rrData(dynamic_cast<CSRNumericTableIface *>(_ntData)), rrClass(_ntClass), n_ci(local_n_ci)
    {}

    size_t getBlockSize(size_t n) { return n; }

    Status addData(size_t nStart, size_t blockSize)
    {
        rrData.next(nStart, blockSize);
        DAAL_CHECK_BLOCK_STATUS(rrData);
        rrClass.next(nStart, blockSize);
        DAAL_CHECK_BLOCK_STATUS(rrClass);

        const algorithmFPType * data = rrData.values();
        const size_t * colIdx        = rrData.cols();
        const size_t * rowIdx        = rrData.rows();
        const int * predefClass      = rrClass.get();

        size_t k = 0;

        for (size_t j = 0; j < blockSize; j++)
        {
            size_t cl = predefClass[j];
            DAAL_ASSERT(cl < _c);
            size_t jn = rowIdx[j + 1] - rowIdx[j];

            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < jn; i++)
            {
                size_t col = colIdx[k + i] - 1;

                n_ci[cl * _p + col] += data[k + i];
            }

            k += jn;
        }

        return Status();
    }
};

template <typename algorithmFPType, Method method, CpuType cpu>
Status collectCounters(const Parameter * nbPar, NumericTable * ntData, NumericTable * ntClass, algorithmFPType * n_c, algorithmFPType * n_ci)
{
    size_t p = ntData->getNumberOfColumns();
    size_t n = ntData->getNumberOfRows();
    size_t c = nbPar->nClasses;

    daal::tls<algorithmFPType *> tls_n_ci([=]() -> algorithmFPType * { return _CALLOC_<algorithmFPType, cpu>(p * c); });

    SafeStatus safeStat;
    daal::threader_for_blocked(n, n, [=, &tls_n_ci, &safeStat](algorithmFPType j0, algorithmFPType jn) {
        algorithmFPType * local_n_ci = tls_n_ci.local();
        DAAL_CHECK_THR(local_n_ci, ErrorMemoryAllocationFailed);

        localDataCollector<algorithmFPType, method, cpu> ldc(p, c, ntData, ntClass, local_n_ci);

        algorithmFPType block_size = ldc.getBlockSize(jn);
        int i;

        for (i = 0; i + block_size < jn + 1; i += block_size)
        {
            safeStat |= ldc.addData(j0 + i, block_size);
        }

        if (i != jn)
        {
            safeStat |= ldc.addData(j0 + i, jn - i);
        }
    });

    tls_n_ci.reduce([=](algorithmFPType * v) {
        if (!v) return;

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (size_t j = 0; j < c; j++)
        {
            for (size_t i = 0; i < p; i++)
            {
                n_ci[j * p + i] += v[j * p + i];
                n_c[j] += v[j * p + i];
            }
        }
        _FREE_<algorithmFPType, cpu>(v);
    });

    return safeStat.detach();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status mergeModels(const Parameter * nbPar, size_t p, size_t nModels, PartialModel * const * models, algorithmFPType * n_c, algorithmFPType * n_ci,
                   size_t & merged_n)
{
    size_t c = nbPar->nClasses;

    ReadRows<algorithmFPType, cpu> rrC;
    ReadRows<algorithmFPType, cpu> rrCi;

    for (size_t i = 0; i < nModels; i++)
    {
        const algorithmFPType * in_n_c = rrC.set(models[i]->getClassSize().get(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(rrC);
        const algorithmFPType * in_n_ci = rrCi.set(models[i]->getClassGroupSum().get(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(rrCi);

        PRAGMA_FORCE_SIMD
        for (size_t j = 0; j < c; j++)
        {
            n_c[j] += in_n_c[j];
        }

        PRAGMA_FORCE_SIMD
        for (size_t j = 0; j < p * c; j++)
        {
            n_ci[j] += in_n_ci[j];
        }

        merged_n += models[i]->getNObservations();
    }

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status fillModel(const Parameter * nbPar, size_t p, algorithmFPType * n_c, algorithmFPType * n_ci, Model * rMdl)
{
    size_t c = nbPar->nClasses;

    WriteOnlyRows<algorithmFPType, cpu> wrLogP(*rMdl->getLogP(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrLogP);
    WriteOnlyRows<algorithmFPType, cpu> wrLogTheta(*rMdl->getLogTheta(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrLogTheta);
    WriteOnlyRows<algorithmFPType, cpu> wrAuxTable(*rMdl->getAuxTable(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrAuxTable);

    algorithmFPType * log_p     = wrLogP.get();
    algorithmFPType * log_theta = wrLogTheta.get();
    algorithmFPType * aux_table = wrAuxTable.get();

    if (!nbPar->priorClassEstimates.get())
    {
        algorithmFPType log_p_const = -daal::internal::MathInst<algorithmFPType, cpu>::sLog((algorithmFPType)c);

        for (size_t j = 0; j < c; j++)
        {
            log_p[j] = log_p_const;
        }
    }
    else
    {
        ReadRows<algorithmFPType, cpu> rrPriorClassEstimates(*nbPar->priorClassEstimates, 0, c);
        DAAL_CHECK_BLOCK_STATUS(rrPriorClassEstimates);
        const algorithmFPType * pe = rrPriorClassEstimates.get();

        daal::internal::MathInst<algorithmFPType, cpu>::vLog(c, pe, log_p);
    }

    if (!nbPar->alpha.get())
    {
        algorithmFPType alpha_i = 1;
        algorithmFPType alpha   = p * alpha_i;

        for (size_t j = 0; j < c; j++)
        {
            algorithmFPType denominator = (algorithmFPType)1.0 / (n_c[j] + alpha);

            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < p; i++)
            {
                log_theta[j * p + i] = (n_ci[j * p + i] + alpha_i) * denominator;
            }
            daal::internal::MathInst<algorithmFPType, cpu>::vLog(p, log_theta + j * p, log_theta + j * p);
        }
    }
    else
    {
        ReadRows<algorithmFPType, cpu> rrAlphaI(*nbPar->alpha, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(rrAlphaI);
        const algorithmFPType * alpha_i = rrAlphaI.get();

        algorithmFPType alpha = 0;
        for (size_t i = 0; i < p; i++)
        {
            alpha += alpha_i[i];
        }

        for (size_t j = 0; j < c; j++)
        {
            algorithmFPType denominator = (algorithmFPType)1.0 / (n_c[j] + alpha);

            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < p; i++)
            {
                log_theta[j * p + i] = (n_ci[j * p + i] + alpha_i[i]) * denominator;
            }
            daal::internal::MathInst<algorithmFPType, cpu>::vLog(p, log_theta + j * p, log_theta + j * p);
        }
    }

    for (size_t j = 0; j < c; j++)
    {
        for (size_t i = 0; i < p; i++)
        {
            aux_table[j * p + i] = log_theta[j * p + i] + log_p[j];
        }
    }

    rMdl->setNFeatures(p);

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NaiveBayesBatchTrainKernel<algorithmFPType, method, cpu>::compute(const NumericTable * data, const NumericTable * labels,
                                                                                   Model * mdl, const Parameter * nbPar)
{
    NumericTable * ntData  = const_cast<NumericTable *>(data);
    NumericTable * ntClass = const_cast<NumericTable *>(labels);

    size_t p = ntData->getNumberOfColumns();
    size_t c = nbPar->nClasses;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, c, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, c, p);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, c * p, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> t_n_c(c);
    TArray<algorithmFPType, cpu> t_n_ci(c * p);
    algorithmFPType * n_c  = t_n_c.get();
    algorithmFPType * n_ci = t_n_ci.get();

    if (n_c == 0 || n_ci == 0)
    {
        return Status(ErrorMemoryAllocationFailed);
    }

    PRAGMA_FORCE_SIMD
    for (size_t j = 0; j < c; j++)
    {
        n_c[j] = 0;
    }

    PRAGMA_FORCE_SIMD
    for (size_t j = 0; j < p * c; j++)
    {
        n_ci[j] = 0;
    }
    Status s = collectCounters<algorithmFPType, method, cpu>(nbPar, ntData, ntClass, n_c, n_ci);

    if (!s) return s;

    return fillModel<algorithmFPType, method, cpu>(nbPar, p, n_c, n_ci, mdl);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NaiveBayesOnlineTrainKernel<algorithmFPType, method, cpu>::compute(const NumericTable * data, const NumericTable * labels,
                                                                                    PartialModel * mdl, const Parameter * nbPar)
{
    NumericTable * ntData  = const_cast<NumericTable *>(data);
    NumericTable * ntClass = const_cast<NumericTable *>(labels);

    size_t c = nbPar->nClasses;

    Status s;

    WriteRows<algorithmFPType, cpu> wrC(*mdl->getClassSize(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrC);
    WriteRows<algorithmFPType, cpu> wrCi(*mdl->getClassGroupSum(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrCi);

    s |= collectCounters<algorithmFPType, method, cpu>(nbPar, ntData, ntClass, wrC.get(), wrCi.get());

    size_t n = ntData->getNumberOfRows();

    mdl->setNObservations(mdl->getNObservations() + n);
    mdl->setNFeatures(ntData->getNumberOfColumns());

    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NaiveBayesOnlineTrainKernel<algorithmFPType, method, cpu>::finalizeCompute(PartialModel * pMdl, Model * rMdl,
                                                                                            const Parameter * nbPar)
{
    size_t c = nbPar->nClasses;

    WriteRows<algorithmFPType, cpu> wrC(*pMdl->getClassSize(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrC);
    WriteRows<algorithmFPType, cpu> wrCi(*pMdl->getClassGroupSum(), 0, c);
    DAAL_CHECK_BLOCK_STATUS(wrCi);

    return fillModel<algorithmFPType, method, cpu>(nbPar, pMdl->getNFeatures(), wrC.get(), wrCi.get(), rMdl);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NaiveBayesDistributedTrainKernel<algorithmFPType, method, cpu>::merge(size_t nModels, PartialModel * const * inPMdls,
                                                                                       PartialModel * outPMdl, const Parameter * nbPar)
{
    size_t c = nbPar->nClasses;
    size_t p = outPMdl->getNFeatures();

    size_t merged_n = 0;

    Status s;

    if (outPMdl->getNObservations() == 0)
    {
        WriteOnlyRows<algorithmFPType, cpu> wrC(*outPMdl->getClassSize(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(wrC);
        WriteOnlyRows<algorithmFPType, cpu> wrCi(*outPMdl->getClassGroupSum(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(wrCi);

        algorithmFPType * n_c  = wrC.get();
        algorithmFPType * n_ci = wrCi.get();

        for (size_t j = 0; j < c; j++)
        {
            n_c[j] = 0;
        }

        for (size_t j = 0; j < p * c; j++)
        {
            n_ci[j] = 0;
        }

        s = mergeModels<algorithmFPType, method, cpu>(nbPar, p, nModels, inPMdls, n_c, n_ci, merged_n);
    }
    else
    {
        WriteRows<algorithmFPType, cpu> wrC(*outPMdl->getClassSize(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(wrC);
        WriteRows<algorithmFPType, cpu> wrCi(*outPMdl->getClassGroupSum(), 0, c);
        DAAL_CHECK_BLOCK_STATUS(wrCi);

        s = mergeModels<algorithmFPType, method, cpu>(nbPar, p, nModels, inPMdls, wrC.get(), wrCi.get(), merged_n);
    }

    outPMdl->setNObservations(outPMdl->getNObservations() + merged_n);

    return s;
}

} // namespace internal
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal

#endif
