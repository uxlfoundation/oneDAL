/* file: qr_dense_default_batch_impl.i */
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

#ifndef __QR_KERNEL_BATCH_IMPL_I__
#define __QR_KERNEL_BATCH_IMPL_I__

#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include "src/algorithms/qr/qr_dense_default_impl.i"

#include "src/threading/threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{
#include "src/algorithms/qr/qr_dense_default_pcl_impl.i"

/**
 *  \brief Kernel for QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                            const daal::algorithms::Parameter * par)
{
    NumericTable * ntAi = const_cast<NumericTable *>(a[0]);

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();
    const size_t t = threader_get_threads_number();

    if (m >= 2 * n)
    {
        if ((m > n * t) && (n > 10) && (!(n >= 200 && m <= 100000)))
            return QRBatchKernel<algorithmFPType, method, cpu>::compute_pcl(na, a, nr, r, par);
        return QRBatchKernel<algorithmFPType, method, cpu>::compute_thr(na, a, nr, r, par);
    }
    return QRBatchKernel<algorithmFPType, method, cpu>::compute_seq(na, a, nr, r, par);
}

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_seq(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                const daal::algorithms::Parameter * par)
{
    NumericTable * ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable * ntRi = const_cast<NumericTable *>(r[1]);

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, m);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * m, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> QiTPtr(n * m);
    algorithmFPType * QiT = QiTPtr.get();
    TArray<algorithmFPType, cpu> RiTPtr(n * n);
    algorithmFPType * RiT = RiTPtr.get();
    DAAL_CHECK(QiT && RiT, ErrorMemoryAllocationFailed);

    //copy Ai to QiT, transposed
    {
        ReadRows<algorithmFPType, cpu, NumericTable> aiBlock(ntAi, 0, m);
        DAAL_CHECK_BLOCK_STATUS(aiBlock);
        const algorithmFPType * Ai = aiBlock.get();
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                QiT[i * m + j] = Ai[i + j * n];
            }
        }
    }

    DAAL_INT ldAi = m;
    DAAL_INT ldRi = n;
    const auto ec = compute_QR_on_one_node<algorithmFPType, cpu>(m, n, QiT, ldAi, RiT, ldRi);
    DAAL_CHECK_STATUS_VAR(ec);

    //copy QiT to Qi, transposed
    {
        NumericTable * ntQi = const_cast<NumericTable *>(r[0]);
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> qiBlock(ntQi, 0, m); /* Qi = Qin[m][n] */
        DAAL_CHECK_BLOCK_STATUS(qiBlock);
        algorithmFPType * Qi = qiBlock.get();
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }
    }

    //copy RiT to Ri, transposed
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> riBlock(ntRi, 0, n); /* Ri = Ri [n][n] */
        DAAL_CHECK_BLOCK_STATUS(riBlock);
        algorithmFPType * Ri = riBlock.get();
        for (size_t i = 0; i < n; i++)
        {
            size_t j = 0;
            for (; j <= i; j++)
            {
                Ri[i + j * n] = RiT[i * n + j];
            }
            for (; j < n; j++)
            {
                Ri[i + j * n] = 0.0;
            }
        }
    }
    return Status();
}

/* Max number of blocks depending on arch */
#if (__CPUID__(DAAL_CPU) >= __avx512__)
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif

/*
    Algorithm for parallel QR computation:
    -------------------------------------
    A[m,n] input matrix to be factorized by output Q[m,n] and R[n,n]

    1st step:
    Split A[m,n] matrix to 'b' blocks -> a1[m1,n],a2[m2,]...ab[mb,n]
    Compute QR decomposition for each block in threads:
                               a1[m1,n] -> q1[m1,n] , r1[n,n] ... ab[mb,n] -> qb[mb,n] and rb[n,n]

    2nd step:
    Concatenate r1[n,n] , r2[n,n] ... rb[n,n] into one matrix B[n*b,n]
    Compute QR decomposition for B[n*b,n] -> P[n*b,n] , R[n,n]. R - resulted matrix

    3rd step: Split P[n*b,n] matrix to 'b' blocks -> p1[n,n],p2[n,n]...pb[n,n]
    Multiply by q1..qb matrices from 1st step using GEMM for each block in threads:
                               q1[m1,n] * p1[n,n] -> q'1[m1,n] ... qb[mb,n] * pb[n,n] -> q'b[mb,n]
    Concatenate q'1[m1,n]...q'b[mb,n] into one resulted Q[m,n]  matrix.

    Notice: before and after QR and GEMM computations matrices need to be transposed
*/
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_thr(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                const daal::algorithms::Parameter * par)
{
    NumericTable * ntA_input  = const_cast<NumericTable *>(a[0]);
    NumericTable * ntQ_output = const_cast<NumericTable *>(r[0]);
    NumericTable * ntR_output = const_cast<NumericTable *>(r[1]);

    const size_t n = ntA_input->getNumberOfColumns();
    const size_t m = ntA_input->getNumberOfRows();

    size_t rows = m;
    size_t cols = n;

    /* Getting real pointers to output array */
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkQ_output(ntQ_output, 0, m);
    DAAL_CHECK_BLOCK_STATUS(bkQ_output);
    algorithmFPType * Q_output = bkQ_output.get();

    /* Block size calculation (empirical) */
    const int bshift = (rows <= 10000) ? 11 : 12;
    int bsize        = ((rows * cols) >> bshift) & (~0xf);
    bsize            = (bsize < 200) ? 200 : bsize;

    /*
    Calculate sizes:
    blocks     = number of blocks,
    brows      = number of rows in blocks,
    brows_last = number of rows in last block,
    */
    size_t def_min_brows = rows / DEF_MAX_BLOCKS;                           // min block size
    size_t brows         = (rows > bsize) ? bsize : rows;                   /* brows cannot be less than rows */
    brows                = (brows < cols) ? cols : brows;                   /* brows cannot be less than cols */
    brows                = (brows < def_min_brows) ? def_min_brows : brows; /* brows cannot be less than n/DEF_MAX_BLOCKS */
    size_t blocks        = rows / brows;

    size_t brows_last = brows + (rows - blocks * brows); /* last block is generally biggest */

    size_t len = blocks * n * n;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, blocks);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, len, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> R_buffPtr(n * n);
    algorithmFPType * R_buff = R_buffPtr.get();
    DAAL_CHECK(R_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> RT_buffPtr(len);
    algorithmFPType * RT_buff = RT_buffPtr.get();
    DAAL_CHECK(RT_buff, ErrorMemoryAllocationFailed);

    SafeStatus safeStat;
    /* Step1: calculate QR on local nodes */
    /* ================================== */
    {
        /* Getting real pointers to input array */
        ReadRows<algorithmFPType, cpu, NumericTable> bkA_input(ntA_input, 0, m);
        DAAL_CHECK_BLOCK_STATUS(bkA_input);
        const algorithmFPType * A_input = bkA_input.get();

        daal::threader_for(blocks, blocks, [=, &safeStat](int k) {
            const algorithmFPType * A_block = A_input + k * brows * cols;
            algorithmFPType * Q_block       = Q_output + k * brows * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            const size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
            const size_t cols_local  = cols;

            TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_local = QT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
            algorithmFPType * RT_local = RT_local_Arr.get();

            DAAL_CHECK_THR(QT_local && RT_local, ErrorMemoryAllocationFailed);

            /* Get transposed Q from A */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_FORCE_SIMD
                for (size_t j = 0; j < brows_local; j++)
                {
                    QT_local[i * brows_local + j] = A_block[i + j * cols_local];
                }
            }

            /* Call QR on local nodes */
            const auto ec = compute_QR_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local);
            DAAL_CHECK_STATUS_THR(ec);

            /* Transpose Q */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_FORCE_SIMD
                for (size_t j = 0; j < brows_local; j++)
                {
                    Q_block[i + j * cols_local] = QT_local[i * brows_local + j];
                }
            }

            /* Transpose R and zero lower values */
            for (size_t i = 0; i < cols_local; i++)
            {
                size_t j;
                PRAGMA_FORCE_SIMD
                for (j = 0; j <= i; j++)
                {
                    RT_buff[k * cols_local + i * cols_local * blocks + j] = RT_local[i * cols_local + j];
                }

                service_memset<algorithmFPType, cpu>(&RT_buff[k * cols_local + i * cols_local * blocks + i + 1], 0.0, cols_local - i - 1);
            }
        });
    }

    DAAL_CHECK_SAFE_STATUS();

    /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */
    /* ============================================================== */

    /* Call QR on master node for RB */
    const auto ec = compute_QR_on_one_node_seq<algorithmFPType, cpu>(cols * blocks, cols, RT_buff, cols * blocks, R_buff, cols);
    DAAL_CHECK_STATUS_VAR(ec);

    /* Transpose R */
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkR_output(ntR_output, 0, n);
        DAAL_CHECK_BLOCK_STATUS(bkR_output);
        algorithmFPType * R_output = bkR_output.get();
        for (size_t i = 0; i < cols; i++)
        {
            PRAGMA_FORCE_SIMD
            for (size_t j = 0; j < cols; j++)
            {
                R_output[i + j * cols] = R_buff[i * cols + j];
            }
        }
    }
    /* Step3: calculate Q by merging Q*RB */
    /* ================================== */

    daal::threader_for(blocks, blocks, [=, &safeStat](int k) {
        algorithmFPType * Q_block = Q_output + k * brows * cols;

        /* Last block size brows_last (generally larger than other blocks) */
        size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
        size_t cols_local  = cols;

        TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
        algorithmFPType * QT_local = QT_local_Arr.get();
        TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
        algorithmFPType * RT_local = RT_local_Arr.get();
        TArrayScalable<algorithmFPType, cpu> QT_result_local_Arr(cols_local * brows_local);
        algorithmFPType * QT_result_local = QT_result_local_Arr.get();

        DAAL_CHECK_THR(QT_local && QT_result_local && RT_local, ErrorMemoryAllocationFailed);

        /* Transpose RB */
        for (size_t i = 0; i < cols_local; i++)
        {
            PRAGMA_FORCE_SIMD
            for (size_t j = 0; j < cols_local; j++)
            {
                RT_local[j * cols_local + i] = RT_buff[j * cols_local * blocks + k * cols_local + i];
            }
        }

        /* Transpose Q to QT */
        for (size_t i = 0; i < cols_local; i++)
        {
            PRAGMA_FORCE_SIMD
            for (size_t j = 0; j < brows_local; j++)
            {
                QT_local[i * brows_local + j] = Q_block[i + j * cols_local];
            }
        }

        /* Call GEMMs to multiply Q*R */
        compute_gemm_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local,
                                                           brows_local);

        /* Transpose result Q */
        for (size_t i = 0; i < cols_local; i++)
        {
            PRAGMA_FORCE_SIMD
            for (size_t j = 0; j < brows_local; j++)
            {
                Q_block[i + j * cols_local] = QT_result_local[i * brows_local + j];
            }
        }
    });

    return safeStat.detach();
}

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_pcl(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                const daal::algorithms::Parameter * par)
{
    NumericTable * ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable * ntQi = const_cast<NumericTable *>(r[0]);
    NumericTable * ntRi = const_cast<NumericTable *>(r[1]);

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();

    ReadRows<algorithmFPType, cpu, NumericTable> aiBlock(ntAi, 0, m);
    DAAL_CHECK_BLOCK_STATUS(aiBlock);
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> qiBlock(ntQi, 0, m);
    DAAL_CHECK_BLOCK_STATUS(qiBlock);
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> riBlock(ntRi, 0, n);
    DAAL_CHECK_BLOCK_STATUS(riBlock);

    const services::ErrorID ec = (services::ErrorID)qr_pcl<algorithmFPType, cpu>(aiBlock.get(), m, n, qiBlock.get(), riBlock.get());
    DAAL_CHECK(!ec, services::ErrorID(ec));

    return Status();
}

} // namespace internal
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
