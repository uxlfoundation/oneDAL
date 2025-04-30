/* file: naivebayes_predict_fast_impl.i */
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
//  Implementation of Fast method for assign K-means algorithm.
//
//  Based on paper: Tackling the Poor Assumptions of Naive Bayes Text Classifiers
//  Url: http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "data_management/data/csr_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_spblas.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#if (__CPUID__(DAAL_CPU) >= __avx512__)

    #if (__FPTYPE__(DAAL_FPTYPE) == __double__)

        #define _BLOCKSIZE_ (((p <= 100) && (p < c)) ? ((p <= 20) ? 32 : 64) : 256)

    #else /* float */

        #define _BLOCKSIZE_ ((c > 100) ? 128 : 256)

    #endif

    #define _CALLOC_ service_scalable_calloc
    #define _FREE_   service_scalable_free

#else

    #define _BLOCKSIZE_ (128)
    #define _CALLOC_    service_calloc
    #define _FREE_      service_free

#endif

using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
struct methodSpecific
{};

template <typename algorithmFPType, CpuType cpu>
struct methodSpecific<defaultDense, algorithmFPType, cpu>
{
    static services::Status getPredictionData(const algorithmFPType * aux_table, NumericTable * ntData, size_t n0, size_t n, size_t p, size_t c,
                                              int * classes, algorithmFPType * buff);
};

template <typename algorithmFPType, CpuType cpu>
struct methodSpecific<fastCSR, algorithmFPType, cpu>
{
    static services::Status getPredictionData(const algorithmFPType * aux_table, NumericTable * ntData, size_t n0, size_t n, size_t p, size_t c,
                                              int * classes, algorithmFPType * buff);
};

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NaiveBayesPredictKernel<algorithmFPType, method, cpu>::compute(const NumericTable * a, const Model * m, NumericTable * r,
                                                                                const Parameter * parameter)
{
    NumericTable * ntData  = const_cast<NumericTable *>(a);
    Model * mdl            = const_cast<Model *>(m);
    NumericTable * ntClass = r;

    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t c = parameter->nClasses;

    NumericTable * ntAuxTable = mdl->getAuxTable().get();

    ReadRows<algorithmFPType, cpu> rrAuxTable(ntAuxTable, 0, c);
    DAAL_CHECK_BLOCK_STATUS(rrAuxTable);
    const algorithmFPType * aux_table = rrAuxTable.get();

    size_t blockSizeDeafult = _BLOCKSIZE_;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks * blockSizeDeafult != n);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSizeDeafult, c);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSizeDeafult * c, sizeof(algorithmFPType));

    daal::tls<algorithmFPType *> mkl_buff([=]() -> algorithmFPType * { return _CALLOC_<algorithmFPType, cpu>(blockSizeDeafult * c); });

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [=, &mkl_buff, &safeStat](int k) {
        algorithmFPType * buff = mkl_buff.local();
        DAAL_CHECK_THR(buff, ErrorMemoryAllocationFailed);

        size_t jn = blockSizeDeafult;
        if (k == nBlocks - 1)
        {
            jn = n - k * blockSizeDeafult;
        }
        size_t j0 = k * blockSizeDeafult;

        WriteOnlyRows<int, cpu> wrClassesBlock(ntClass, j0, jn);
        DAAL_CHECK_BLOCK_STATUS_THR(wrClassesBlock);
        int * classes = wrClassesBlock.get();

        safeStat |= methodSpecific<method, algorithmFPType, cpu>::getPredictionData(aux_table, ntData, j0, jn, p, c, classes, buff);
    });

    mkl_buff.reduce([=](algorithmFPType * v) -> void {
        if (v) _FREE_<algorithmFPType, cpu>(v);
    });

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
services::Status methodSpecific<defaultDense, algorithmFPType, cpu>::getPredictionData(const algorithmFPType * aux_table, NumericTable * ntData,
                                                                                       size_t n0, size_t n, size_t p, size_t c, int * classes,
                                                                                       algorithmFPType * buff)
{
    ReadRows<algorithmFPType, cpu> rrData(ntData, n0, n);
    DAAL_CHECK_BLOCK_STATUS(rrData);
    const algorithmFPType * data = rrData.get();

    {
        const char transa           = 't';
        const char transb           = 'n';
        const DAAL_INT _m           = c;
        const DAAL_INT _n           = n;
        const DAAL_INT _k           = p;
        const algorithmFPType alpha = 1.0;
        const DAAL_INT lda          = p;
        const DAAL_INT ldy          = p;
        const algorithmFPType beta  = 0.0;
        const DAAL_INT ldaty        = c;

        BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, aux_table, &lda, data, &ldy, &beta, buff, &ldaty);
    }

    for (size_t j = 0; j < n; j++)
    {
        int max_c                 = 0;
        algorithmFPType max_c_val = -(services::internal::MaxVal<algorithmFPType>::get());

        PRAGMA_FORCE_SIMD
        for (size_t cl = 0; cl < c; cl++)
        {
            algorithmFPType val = buff[j * c + cl];

            if (val > max_c_val)
            {
                max_c_val = val;
                max_c     = cl;
            }
        }

        classes[j] = max_c;
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status methodSpecific<fastCSR, algorithmFPType, cpu>::getPredictionData(const algorithmFPType * aux_table, NumericTable * ntData, size_t n0,
                                                                                  size_t n, size_t p, size_t c, int * classes, algorithmFPType * buff)
{
    CSRNumericTableIface * ntCSRData = dynamic_cast<CSRNumericTableIface *>(ntData);
    ReadRowsCSR<algorithmFPType, cpu> rrData(ntCSRData, n0, n);
    DAAL_CHECK_BLOCK_STATUS(rrData);

    const algorithmFPType * values = rrData.values();
    const size_t * colIdx          = rrData.cols();
    const size_t * rowIdx          = rrData.rows();

    {
        const char transa           = 'n';
        const DAAL_INT _n           = n;
        const DAAL_INT _p           = p;
        const DAAL_INT _c           = c;
        const algorithmFPType alpha = 1.0;
        const algorithmFPType beta  = 0.0;
        const char matdescra[6]     = { 'G', 0, 0, 'F', 0, 0 };

        SpBlasInst<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra, values, (DAAL_INT *)colIdx, (DAAL_INT *)rowIdx,
                                                  aux_table, &_p, &beta, buff, &_n);
    }

    for (size_t j = 0; j < n; j++)
    {
        int max_c                 = 0;
        algorithmFPType max_c_val = -(services::internal::MaxVal<algorithmFPType>::get());

        PRAGMA_FORCE_SIMD
        for (size_t cl = 0; cl < c; cl++)
        {
            algorithmFPType val = buff[j + cl * n];

            if (val > max_c_val)
            {
                max_c_val = val;
                max_c     = cl;
            }
        }

        classes[j] = max_c;
    }

    return services::Status();
}

} // namespace internal
} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
