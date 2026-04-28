/* file: qr_dense_default_distr_impl.i */
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
//  Implementation of qrs
//--
*/

#ifndef __QR_KERNEL_DISTR_IMPL_I__
#define __QR_KERNEL_DISTR_IMPL_I__

#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include "src/algorithms/qr/qr_dense_default_impl.i"

#include "src/threading/threading.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{
/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRDistributedStep2Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                       NumericTable * r0, NumericTable * r[], const daal::algorithms::Parameter * par,
                                                                       data_management::KeyValueDataCollection * inCollection)
{
    const NumericTable * ntAux2_0 = a[0];

    size_t nBlocks = na;

    const size_t n   = ntAux2_0->getNumberOfColumns(); /* size of observations block */
    const size_t nxb = n * nBlocks;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nxb);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nxb, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> aAux2T(n * nxb);
    algorithmFPType * Aux2T = aAux2T.get();
    TArray<algorithmFPType, cpu> aRT(n * n);
    algorithmFPType * RT = aRT.get();
    DAAL_CHECK(Aux2T && RT, ErrorMemoryAllocationFailed);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, 1, [=, &safeStat](int k) {
        ReadRows<algorithmFPType, cpu, NumericTable> mtAux2(*const_cast<NumericTable *>(a[k]), 0, n);
        DAAL_CHECK_BLOCK_STATUS_THR(mtAux2);
        const algorithmFPType * Aux2 = mtAux2.get(); /* Aux2  [nxb][n] */
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                Aux2T[j * nxb + k * n + i] = Aux2[i * n + j];
            }
        }
    });
    DAAL_CHECK_SAFE_STATUS();

    {
        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * n; i++)
        {
            RT[i] = 0.0;
        }
    }

    const auto ec = compute_QR_on_one_node<algorithmFPType, cpu>(nxb, n, Aux2T, nxb, RT, n);
    DAAL_CHECK_STATUS_VAR(ec);

    daal::threader_for(nBlocks, 1, [=, &safeStat](int k) {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> mtAux3(r[k], 0, n);
        DAAL_CHECK_BLOCK_STATUS_THR(mtAux3);
        algorithmFPType * Aux3 = mtAux3.get(); /* Aux2  [nxb][n] */
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                Aux3[i * n + j] = Aux2T[j * nxb + k * n + i];
            }
        }
    });
    DAAL_CHECK_SAFE_STATUS();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> mtR(r0, 0, n); /* V[n][n] */
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * R = mtR.get();

    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            R[i + j * n] = RT[i * n + j];
        }
    }

    if (inCollection)
    {
        inCollection->clear();
    }

    return Status();
}

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRDistributedStep3Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                       NumericTable * r[], const daal::algorithms::Parameter * par)
{
    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    for (size_t k = 0; k < nBlocks; k++)
    {
        const size_t n = a[k]->getNumberOfColumns();
        const size_t m = a[k]->getNumberOfRows();

        ReadRows<algorithmFPType, cpu, NumericTable> mtAux1i(*const_cast<NumericTable *>(a[k]), 0, m);
        DAAL_CHECK_BLOCK_STATUS(mtAux1i);
        ReadRows<algorithmFPType, cpu, NumericTable> mtAux3i(*const_cast<NumericTable *>(a[k + nBlocks]), 0, n);
        DAAL_CHECK_BLOCK_STATUS(mtAux3i);
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> mtQi(r[0], mCalculated, m);
        DAAL_CHECK_BLOCK_STATUS(mtQi);

        const algorithmFPType * Aux1i = mtAux1i.get(); /* Aux1i = Qin[m][n] */
        const algorithmFPType * Aux3i = mtAux3i.get(); /* Aux3i = Ri [n][n] */
        algorithmFPType * Qi          = mtQi.get();    /*         Qi [m][n] */

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, m);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * m, sizeof(algorithmFPType));
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));

        TArray<algorithmFPType, cpu> QiT_Arr(n * m);
        algorithmFPType * QiT = QiT_Arr.get();
        TArray<algorithmFPType, cpu> Aux1iT_Arr(n * m);
        algorithmFPType * Aux1iT = Aux1iT_Arr.get();
        TArray<algorithmFPType, cpu> Aux3iT_Arr(n * n);
        algorithmFPType * Aux3iT = Aux3iT_Arr.get();

        DAAL_CHECK(QiT && Aux1iT && Aux3iT, ErrorMemoryAllocationFailed);

        for (auto i = 0; i < n; i++)
        {
            for (auto j = 0; j < m; j++)
            {
                Aux1iT[i * m + j] = Aux1i[i + j * n];
            }
        }
        for (auto i = 0; i < n; i++)
        {
            for (auto j = 0; j < n; j++)
            {
                Aux3iT[i * n + j] = Aux3i[i + j * n];
            }
        }

        DAAL_INT ldAux1i = m;
        DAAL_INT ldAux3i = n;
        DAAL_INT ldQi    = m;

        compute_gemm_on_one_node<algorithmFPType, cpu>(m, n, Aux1iT, ldAux1i, Aux3iT, ldAux3i, QiT, ldQi);

        for (auto i = 0; i < n; i++)
        {
            for (auto j = 0; j < m; j++)
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }

        mCalculated += m;
    }
    return Status();
}

} // namespace internal
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
