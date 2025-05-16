/* file: pca_dense_svd_batch_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_IMPL_I__
#define __PCA_DENSE_SVD_BATCH_IMPL_I__

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status PCASVDBatchKernel<algorithmFPType, ParameterType, cpu>::compute(InputDataType type, const NumericTablePtr & data,
                                                                                 NumericTable & eigenvalues, NumericTable & eigenvectors)
{
    NumericTablePtr normalizedData;
    if (type == normalizedDataset)
    {
        normalizedData = data;
    }
    else
    {
        services::Status s = normalizeDataset(data, normalizedData);
        if (!s) return s;
    }

    Status s = decompose(normalizedData.get(), eigenvalues, eigenvectors);
    if (s) s = this->scaleSingularValues(eigenvalues, data->getNumberOfRows());
    return s;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status PCASVDBatchKernel<algorithmFPType, ParameterType, cpu>::compute(InputDataType type, NumericTable & data,
                                                                                 const ParameterType * parameter, NumericTable & eigenvalues,
                                                                                 NumericTable & eigenvectors, NumericTable & means,
                                                                                 NumericTable & variances)
{
    NumericTable * normalizedData;
    Status status;
    if (type == normalizedDataset)
    {
        normalizedData = &data;
        if (parameter->resultsToCompute & mean)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(means, (algorithmFPType)0));
        }
        if (parameter->resultsToCompute & variance)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(variances, (algorithmFPType)1));
        }
    }
    else
    {
        auto normalizationAlgorithm = parameter->normalization;
        DAAL_CHECK_STATUS(status, normalizationAlgorithm->computeNoThrow());
        normalizedData = normalizationAlgorithm->getResult()->get(normalization::zscore::normalizedData).get();

        if (parameter->resultsToCompute & mean)
        {
            auto & zmeans = *(normalizationAlgorithm->getResult()->get(normalization::zscore::means));
            DAAL_CHECK_STATUS(status, this->copyTable(zmeans, means));
        }

        if (parameter->resultsToCompute & variance)
        {
            auto & zvariances = *(normalizationAlgorithm->getResult()->get(normalization::zscore::variances));
            DAAL_CHECK_STATUS(status, this->copyTable(zvariances, variances));
        }
    }

    DAAL_CHECK_STATUS(status, this->decompose(normalizedData, eigenvalues, eigenvectors));
    if (parameter->doScale)
    {
        DAAL_CHECK_STATUS(status, this->scaleSingularValues(eigenvalues, data.getNumberOfRows()));
    }
    if (parameter->isDeterministic)
    {
        DAAL_CHECK_STATUS(status, this->signFlipEigenvectors(eigenvectors));
    }

    return status;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status PCASVDBatchKernel<algorithmFPType, ParameterType, cpu>::compute(
    InputDataType type, data_management::NumericTable & data, data_management::NumericTable & eigenvectors,
    data_management::NumericTable & singular_values, data_management::NumericTable & means, data_management::NumericTable & variances,
    data_management::NumericTable * eigenvalues, data_management::NumericTable * explained_variances_ratio, const ParameterType * parameter)
{
    Status status;
    this->compute(type, data, parameter, singular_values, eigenvectors, means, variances);
    if (eigenvalues != nullptr && parameter->doScale == false)
    {
        DAAL_CHECK_STATUS(status, this->computeEigenValues(singular_values, *eigenvalues, data.getNumberOfRows()));
    }
    else
    {
        DAAL_CHECK_STATUS(status, this->copyTable(singular_values, *eigenvalues));
    }
    if (explained_variances_ratio != nullptr)
    {
        DAAL_CHECK_STATUS(status, this->computeExplainedVariancesRatio(*eigenvalues, variances, *explained_variances_ratio));
    }
    return status;
}

/********************* tls_data_t class *******************************************************/
template <typename algorithmFPType, CpuType cpu>
struct tls_data_t
{
    DAAL_NEW_DELETE();
    algorithmFPType * mean;
    algorithmFPType * variance;
    algorithmFPType nvectors;

    tls_data_t(size_t nFeatures) : mean(nullptr), variance(nullptr), nvectors(0)
    {
        mean     = service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
        variance = service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
    }

    bool isValid() const { return mean && variance; }

    ~tls_data_t()
    {
        if (mean) service_scalable_free<algorithmFPType, cpu>(mean);
        if (variance) service_scalable_free<algorithmFPType, cpu>(variance);
    }
};

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status PCASVDBatchKernel<algorithmFPType, ParameterType, cpu>::normalizeDataset(const data_management::NumericTablePtr & data,
                                                                                          NumericTablePtr & normalizedData)
{
    using data_management::NumericTable;
    using data_management::HomogenNumericTable;
    using daal::internal::HomogenNumericTableCPU;

    const size_t nObservations = data->getNumberOfRows();
    const size_t nFeatures     = data->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> block(const_cast<NumericTable &>(*data), 0, nObservations);
    DAAL_CHECK_BLOCK_STATUS(block);
    const algorithmFPType * dataArray = block.get();

    Status status;
    HomogenNumericTableCPU<algorithmFPType, cpu> * normalized = new HomogenNumericTableCPU<algorithmFPType, cpu>(nFeatures, nObservations, status);
    DAAL_CHECK_STATUS_VAR(status);

    normalized->assign(0);
    normalizedData.reset(normalized);

    WriteRows<algorithmFPType, cpu> normalizedBlock(*normalizedData, 0, nObservations);
    DAAL_CHECK_BLOCK_STATUS(normalizedBlock);
    algorithmFPType * normalizedDataArray = normalizedBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, sizeof(algorithmFPType));

    TArrayCalloc<algorithmFPType, cpu> mean_total(nFeatures);
    TArrayCalloc<algorithmFPType, cpu> inv_sigma_total(nFeatures);
    DAAL_CHECK_MALLOC(mean_total.get() && inv_sigma_total.get());

#define _BLOCK_SIZE_ 256
    /* Split rows by blocks, block size cannot be less than nObservations */
    size_t numRowsInBlock = (nObservations > _BLOCK_SIZE_) ? _BLOCK_SIZE_ : nObservations;
    /* Number of blocks */
    size_t numBlocks = nObservations / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + (nObservations - numBlocks * numRowsInBlock);

    SafeStatus safeStat;
    /* TLS data initialization */
    daal::tls<tls_data_t<algorithmFPType, cpu> *> tls_data([=, &safeStat]() {
        auto ptr = new tls_data_t<algorithmFPType, cpu>(nFeatures);
        if (!ptr || !ptr->isValid())
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            delete ptr;
            ptr = nullptr;
        }
        return ptr;
    });

    /* Compute partial means and variances for each block */
    daal::threader_for(numBlocks, numBlocks, [&](int iBlock) {
        struct tls_data_t<algorithmFPType, cpu> * tls_data_local = tls_data.local();
        if (!tls_data_local) return;

        size_t nVectors_local = (iBlock < (numBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
        size_t startRow       = iBlock * numRowsInBlock;

        const algorithmFPType * dataArray_local = dataArray + startRow * nFeatures;
        algorithmFPType * mean_local            = tls_data_local->mean;
        algorithmFPType * variance_local        = tls_data_local->variance;

        for (size_t i = 0; i < nVectors_local; i++)
        {
            const algorithmFPType _invN = algorithmFPType(1.0) / algorithmFPType(tls_data_local->nvectors + 1);

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nFeatures; j++)
            {
                const algorithmFPType delta_local = dataArray_local[i * nFeatures + j] - mean_local[j];

                mean_local[j] += delta_local * _invN;
                variance_local[j] += delta_local * (dataArray_local[i * nFeatures + j] - mean_local[j]);
            }

            tls_data_local->nvectors++;
        }
    });

    algorithmFPType n_current = 0;

    /* Merge mean and variance arrays by blocks */
    tls_data.reduce([=, &inv_sigma_total, &mean_total, &n_current](tls_data_t<algorithmFPType, cpu> * tls_data_local) {
        if (!tls_data_local) return;

        /* loop invariants */
        const algorithmFPType n_local           = tls_data_local->nvectors;
        const algorithmFPType n1_p_n2           = n_current + n_local;
        const algorithmFPType n1_m_n2           = n_current * n_local;
        const algorithmFPType n1_m_n2_o_n1_p_n2 = n1_m_n2 / n1_p_n2;
        const algorithmFPType inv_n1_p_n2       = algorithmFPType(1.0) / (n1_p_n2);
        const algorithmFPType inv_n1_p_n2_m1    = algorithmFPType(1.0) / (n1_p_n2 - algorithmFPType(1.0));

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            const algorithmFPType _delta = tls_data_local->mean[j] - mean_total[j];
            const algorithmFPType _vl    = tls_data_local->variance[j];          /* local variances are not scaled yet */
            const algorithmFPType _vt    = inv_sigma_total[j] * (n_current - 1); /* merged variances are already scaled */

            /* merging variances */
            inv_sigma_total[j] = (_vt + _vl + _delta * _delta * n1_m_n2_o_n1_p_n2) * inv_n1_p_n2_m1;
            /* merging means */
            mean_total[j] = (mean_total[j] * n_current + tls_data_local->mean[j] * n_local) * inv_n1_p_n2;
        }

        /* Increase number of already merged values */
        n_current += n_local;

        delete tls_data_local;
    });
    if (!safeStat) return safeStat.detach();

    /* Convert array of variances to inverse sigma's */
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t j = 0; j < nFeatures; j++)
    {
        if (inv_sigma_total[j]) inv_sigma_total[j] = algorithmFPType(1.0) / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(inv_sigma_total[j]);
    }

    /* Final normalization threaded loop */
    daal::threader_for(numBlocks, numBlocks, [&](int iBlock) {
        size_t nVectors_local = (iBlock < (numBlocks - 1)) ? numRowsInBlock : numRowsInLastBlock;
        size_t startRow       = iBlock * numRowsInBlock;

        const algorithmFPType * dataArray_local = dataArray + startRow * nFeatures;
        algorithmFPType * normDataArray_local   = normalizedDataArray + startRow * nFeatures;

        for (size_t i = 0; i < nVectors_local; i++)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nFeatures; j++)
            {
                normDataArray_local[i * nFeatures + j] = (dataArray_local[i * nFeatures + j] - mean_total[j]) * inv_sigma_total[j];
            }
        }
    });
    return services::Status();
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status PCASVDBatchKernel<algorithmFPType, ParameterType, cpu>::decompose(const NumericTable * normalizedDataTable,
                                                                                   data_management::NumericTable & eigenvalues,
                                                                                   data_management::NumericTable & eigenvectors)
{
    const NumericTable * const * svdInputs = &normalizedDataTable;

    NumericTable * svdResults[3] = { &eigenvalues, nullptr, &eigenvectors };
    svd::Parameter params;
    params.leftSingularMatrix = svd::notRequired;
    daal::algorithms::svd::internal::SVDBatchKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    return svdKernel.compute(1, svdInputs, 3, svdResults, &params);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
