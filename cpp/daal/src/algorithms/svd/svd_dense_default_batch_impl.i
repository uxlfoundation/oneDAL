/* file: svd_dense_default_batch_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_KERNEL_BATCH_IMPL_I__
#define __SVD_KERNEL_BATCH_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include "src/algorithms/svd/svd_dense_default_impl.i"

#include "src/threading/threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
#include "src/algorithms/qr/qr_dense_default_pcl_impl.i"

/**
 *  \brief Kernel for SVD calculation
 */
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                             const daal::algorithms::Parameter * par)
{
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    NumericTable * ntAi = const_cast<NumericTable *>(a[0]);

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();
    const size_t t = threader_get_threads_number();

    if (m < 2 * n) return SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq(na, a, nr, r, svdPar);

    if ((m > n * t) && (n > 10) && (!(n >= 200 && m <= 100000)))
        return SVDBatchKernel<algorithmFPType, method, cpu>::compute_pcl(na, a, nr, r, svdPar);

    return SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr(na, a, nr, r, svdPar);
}

/**
// Helper class that simplifies the logic when the number of elements
// in the results of SVD decomposition differs from the number of elements
// in the resulting numeric tables
*/
template <typename algorithmFPType, CpuType cpu>
class SVDSeqHelper
{
public:
    /**
    // Creates a new instance of the class
    // @param arraySize_ Number of elements in the result of SVD decomposition
    //                   computed with LAPACK XGESVD
    // @param blockSize_ Number of elements in the resulting numeric table returned to the user
    */
    SVDSeqHelper(size_t arraySize_, size_t blockSize_) : arraySize(arraySize_), blockSize(blockSize_), arrayPtr(nullptr), blockPtr(nullptr) {}

    /**
    // Initializes an instance of the class
    // @param numRows        Number of rows in the resulting numeric table returned to the user
    // @param nt             Pointer to the resulting numeric table returned to the user
    // @param setArrayToZero Flag that specifies either to set the corresponding array that stores
    //                       the results of SVD computations to zero before passing it to LAPACK XGESVD
    */
    Status init(size_t numRows, NumericTable * nt, bool setArrayToZero = false)
    {
        /* Initialize the block of results to be returned to the user */
        block.set(nt, 0, numRows);
        DAAL_CHECK_BLOCK_STATUS(block);
        blockPtr = block.get();

        if (arraySize > blockSize)
        {
            /* If the number of elements in the numeric table returned to the user
               is less than the number of elements in the result of SVD decomposition,
               allocate temporary array to store SVD decomposition results */
            array.reset(arraySize);
            arrayPtr = array.get();
            DAAL_CHECK(arrayPtr, ErrorMemoryAllocationFailed);
            if (setArrayToZero)
            {
                const algorithmFPType zero(0.0);
                service_memset<algorithmFPType, cpu>(arrayPtr, zero, arraySize);
            }
        }

        return Status();
    }

    /**
    // Returns pointer to store the results of SVD decomposition computed using LAPACK XGESVD
    */
    algorithmFPType * get() const { return arrayPtr ? arrayPtr : blockPtr; }

    /**
    // If necessary, copies the results of SVD decomposition from temporary array into the data block
    // associated with the resulting numeric table
    */
    Status copyResult()
    {
        if (arrayPtr)
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSize, sizeof(algorithmFPType));
            services::internal::daal_memcpy_s(blockPtr, blockSize * sizeof(algorithmFPType), arrayPtr, blockSize * sizeof(algorithmFPType));
        }
        return Status();
    }

    /**
    // Sets the rest of the resulting numeric table to zero if needed
    */
    void postprocess(size_t meaningfulResultSize)
    {
        if (meaningfulResultSize < blockSize)
        {
            const algorithmFPType zero(0.0);
            service_memset<algorithmFPType, cpu>(blockPtr + meaningfulResultSize, zero, blockSize - meaningfulResultSize);
        }
    }

private:
    TArray<algorithmFPType, cpu> array; /* Temporary array to store the results of SVD decomposition */
    WriteOnlyRows<algorithmFPType, cpu, NumericTable>
        block;                  /* Data block associated with the numeric table that stores the results of SVD decomposition */
    algorithmFPType * arrayPtr; /* Temporary results of SVD decomposition */
    algorithmFPType * blockPtr; /* Results of SVD decomposition associated with the resulting numeric table */
    size_t arraySize;           /* Number of elements in the temporary array needed to store the result of SVD decomposition */
    size_t blockSize;           /* Number of elements in the part of resulting numeric table sufficient to store the results of SVD decomposition */
};

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * const ntA     = const_cast<NumericTable *>(a[0]);
    NumericTable * const ntSigma = const_cast<NumericTable *>(r[0]);

    const size_t n            = ntA->getNumberOfColumns();
    const size_t m            = ntA->getNumberOfRows();
    const size_t minDimension = (n < m) ? n : m;
    const size_t nComponents  = ntSigma->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> Ablock(ntA, 0, m);
    DAAL_CHECK_BLOCK_STATUS(Ablock);
    const algorithmFPType * const A = Ablock.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> Ublock;
    algorithmFPType * U = nullptr; /* Left singular vectors */
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    SVDSeqHelper<algorithmFPType, cpu> VTHelper(n * n, nComponents * n); /* Right singular vectors */

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Ublock.set(r[1], 0, m);
        DAAL_CHECK_BLOCK_STATUS(Ublock);
        U = Ublock.get();
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        Status s = VTHelper.init(nComponents, r[2]);
        DAAL_CHECK_STATUS_VAR(s);
    }

    SVDSeqHelper<algorithmFPType, cpu> SigmaHelper(n, nComponents);
    Status s = SigmaHelper.init(1, r[0], true);
    DAAL_CHECK_STATUS_VAR(s);

    s = compute_svd_on_one_node<algorithmFPType, cpu>(n, m, A, n, SigmaHelper.get(), VTHelper.get(), n, U, m);
    DAAL_CHECK_STATUS_VAR(s);

    s = SigmaHelper.copyResult();
    DAAL_CHECK_STATUS_VAR(s);

    SigmaHelper.postprocess(minDimension);

    s = VTHelper.copyResult();
    DAAL_CHECK_STATUS_VAR(s);

    VTHelper.postprocess(minDimension * n);

    return Status();
}

/* Max number of blocks depending on arch */
#if (__CPUID__(DAAL_CPU) >= __avx512__)
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif

/*
    Algorithm for parallel SVD computation:
    -------------------------------------
    A[m,n] input matrix to be factorized by output U[m,n] and V[n,n]

    1st step:
    Split A[m,n] matrix to 'b' blocks -> a1[m1,n],a2[m2,]...ab[mb,n]
    Compute QR decomposition for each block in threads:
                               a1[m1,n] -> q1[m1,n] , r1[n,n] ... ab[mb,n] -> qb[mb,n] and rb[n,n]

    2nd step:
    Concatenate r1[n,n] , r2[n,n] ... rb[n,n] into one matrix B[n*b,n]
    Compute QR decomposition for B[n*b,n] -> P[n*b,n] , R[n,n]. R - resulted matrix
    Compute SVD decomposition for R, get U, V matrices and Sigma vector.
    V - resulted matrix.
    Multiply U by B matrices.

    3rd step: Split P[n*b,n] matrix to 'b' blocks -> p1[n,n],p2[n,n]...pb[n,n]
    Multiply by q1..qb matrices from 1st step using GEMM for each block in threads:
                               q1[m1,n] * p1[n,n] -> q'1[m1,n] ... qb[mb,n] * pb[n,n] -> q'b[mb,n]
    Concatenate q'1[m1,n]...q'b[mb,n] into one resulted U[m,n]  matrix.

    Notice: before and after QR and GEMM computations matrices need to be transposed
*/
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * ntA_input = const_cast<NumericTable *>(a[0]);
    NumericTable * ntS_output(r[0]);
    NumericTable * ntU_output(r[1]);
    NumericTable * ntV_output(r[2]);

    const size_t n = ntA_input->getNumberOfColumns();
    const size_t m = ntA_input->getNumberOfRows();

    const size_t nComponents = ntS_output->getNumberOfColumns();

    size_t rows = m;
    size_t cols = n;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
    /* Getting real pointers to output array */
    TArray<algorithmFPType, cpu> aS_output(n);
    DAAL_CHECK_MALLOC(aS_output.get());
    algorithmFPType * S_output = aS_output.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkU_output;
    algorithmFPType * U_output;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        U_output = bkU_output.set(ntU_output, 0, m);
        DAAL_CHECK_BLOCK_STATUS(bkU_output);
    }

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkV_output;
    algorithmFPType * V_output;
    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        V_output = bkV_output.set(ntV_output, 0, nComponents);
        DAAL_CHECK_BLOCK_STATUS(bkV_output);
    }

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

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, blocks);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n * blocks, sizeof(algorithmFPType));
    size_t len = blocks * n * n;
    TArrayCalloc<algorithmFPType, cpu> R_buffPtr(len); /* zeroing */
    algorithmFPType * R_buff = R_buffPtr.get();
    DAAL_CHECK(R_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> RT_buffPtr(len);
    algorithmFPType * RT_buff = RT_buffPtr.get();
    DAAL_CHECK(RT_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> Q_buffPtr;
    algorithmFPType * Q_buff = 0;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Q_buff = U_output; /* re-use output U for Q_buff */
    }
    else
    {
        Q_buffPtr.reset(rows * cols);
        Q_buff = Q_buffPtr.get(); /* allocate new Q_buff */
    }

    SafeStatus safeStat;
    /* Step1: calculate QR on local nodes */
    /* ================================== */
    {
        /* Getting real pointers to input array */
        ReadRows<algorithmFPType, cpu, NumericTable> bkA_input(ntA_input, 0, m);
        DAAL_CHECK_BLOCK_STATUS(bkA_input);
        const algorithmFPType * A_input = bkA_input.get();

        daal::threader_for(blocks, blocks, [=, &A_input, &Q_buff, &RT_buff, &safeStat](int k) {
            const algorithmFPType * A_block = A_input + k * brows * cols;
            algorithmFPType * Q_block       = Q_buff + k * brows * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            const size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
            const size_t cols_local  = cols;

            TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_local = QT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
            algorithmFPType * RT_local = RT_local_Arr.get();

            if (!(QT_local && RT_local))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                return;
            }
            /* Get transposed Q from A */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < brows_local; j++)
                {
                    QT_local[i * brows_local + j] = A_block[i + j * cols_local];
                }
            }

            /* Call QR on local nodes */
            const auto ec = compute_QR_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local);
            if (!ec)
            {
                safeStat.add(ec);
                return;
            }

            /* Transpose Q */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < brows_local; j++)
                {
                    Q_block[i + j * cols_local] = QT_local[i * brows_local + j];
                }
            }

            /* Transpose R and zero lower values */
            for (size_t i = 0; i < cols_local; i++)
            {
                size_t j;
                PRAGMA_IVDEP
                for (j = 0; j <= i; j++)
                {
                    RT_buff[k * cols_local + i * cols_local * blocks + j] = RT_local[i * cols_local + j];
                }
                PRAGMA_IVDEP
                for (; j < cols_local; j++)
                {
                    RT_buff[k * cols_local + i * cols_local * blocks + j] = 0.0;
                }
            }
        });
    }
    if (!safeStat) return safeStat.detach();

    /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */
    /* ============================================================== */

    {
        TArray<algorithmFPType, cpu> U_buffPtr(n * n);
        algorithmFPType * U_buff = U_buffPtr.get();
        DAAL_CHECK(U_buff, ErrorMemoryAllocationFailed);

        TArray<algorithmFPType, cpu> V_buffPtr(n * n);
        algorithmFPType * V_buff = V_buffPtr.get();
        DAAL_CHECK(V_buff, ErrorMemoryAllocationFailed);

        /* Call QR on master node for RB */
        const auto ecQr = compute_QR_on_one_node<algorithmFPType, cpu>(cols * blocks, cols, RT_buff, cols * blocks, R_buff, cols);
        if (!ecQr) return ecQr;
        const auto ecSvd = compute_svd_on_one_node<algorithmFPType, cpu>(cols, cols, R_buff, cols, S_output, U_buff, cols, V_buff, cols);
        if (!ecSvd) return ecSvd;

        WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkS_output(ntS_output, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(bkS_output);
        algorithmFPType * tS_output = bkS_output.get();

        for (size_t i = 0; i < nComponents; i++)
        {
            tS_output[i] = S_output[i];
        }

        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            compute_gemm_on_one_node<algorithmFPType, cpu>(cols * blocks, cols, RT_buff, cols * blocks, U_buff, cols, R_buff, cols * blocks);
        }

        if (svdPar->rightSingularMatrix == requiredInPackedForm)
        {
            /* Transpose result R and save to V output */
            for (size_t i = 0; i < cols; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < nComponents; j++)
                {
                    V_output[i + j * cols] = V_buff[i * cols + j];
                }
            }
        }
    }

    /* Step3: calculate Q by merging Q*RB */
    /* ================================== */

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for(blocks, blocks, [=, &RT_buff, &U_output, &Q_buff, &safeStat](int k) {
            algorithmFPType * Q_block  = Q_buff + k * brows * cols;
            algorithmFPType * RT_block = RT_buff + k * cols * cols;
            algorithmFPType * U_block  = U_output + k * brows * cols; /* the same as Q_buff here */

            /* Last block size brows_last (generally larger than other blocks) */
            size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
            size_t cols_local  = cols;

            TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_local = QT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
            algorithmFPType * RT_local = RT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> QT_result_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_result_local = QT_result_local_Arr.get();

            if (!(QT_local && QT_result_local && RT_local))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                return;
            }
            /* Transpose RB */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < cols_local; j++)
                {
                    RT_block[i * cols_local + j] = R_buff[j * cols_local * blocks + k * cols_local + i];
                }
            }

            /* Transpose Q to QT */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < brows_local; j++)
                {
                    QT_local[i * brows_local + j] = Q_block[i + j * cols_local];
                }
            }

            /* Transpose R to RT */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < cols_local; j++)
                {
                    RT_local[i * cols_local + j] = RT_block[i + j * cols_local];
                }
            }

            /* Call GEMMs to multiply Q*R */
            compute_gemm_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local,
                                                               brows_local);

            /* Transpose result Q */
            for (size_t i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (size_t j = 0; j < brows_local; j++)
                {
                    U_block[i + j * cols_local] = QT_result_local[i * brows_local + j];
                }
            }
        });
    } /* if (svdPar->leftSingularMatrix == requiredInPackedForm) */

    return safeStat.detach();
}

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_pcl(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * ntA = const_cast<NumericTable *>(a[0]);
    NumericTable * ntSigma(r[0]);
    NumericTable * ntQ(r[1]);
    NumericTable * ntV(r[2]);

    size_t n           = ntA->getNumberOfColumns();
    size_t m           = ntA->getNumberOfRows();
    size_t nComponents = ntSigma->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> aBlock(ntA, 0, m);
    DAAL_CHECK_BLOCK_STATUS(aBlock);
    algorithmFPType * A = const_cast<algorithmFPType *>(aBlock.get());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> sigmaArray(n);
    DAAL_CHECK_MALLOC(sigmaArray.get());
    algorithmFPType * Sigma = sigmaArray.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> qBlock;
    algorithmFPType * Q = nullptr;

    TArray<algorithmFPType, cpu> vArray;
    algorithmFPType * V = nullptr;

    bool needsU  = false;
    bool needsVT = false;

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Q = qBlock.set(ntQ, 0, m);
        DAAL_CHECK_BLOCK_STATUS(qBlock);
        needsU = true;
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        vArray.reset(n * n);
        DAAL_CHECK_MALLOC(vArray.get());
        V       = vArray.get();
        needsVT = true;
    }

    const services::ErrorID ec = (services::ErrorID)svd_pcl<algorithmFPType, cpu>(A, m, n, needsU, Q, Sigma, needsVT, V);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> sBlock(ntSigma, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(sBlock);
    algorithmFPType * tSigma = sBlock.get();

    for (size_t i = 0; i < nComponents; i++)
    {
        tSigma[i] = Sigma[i];
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> vBlock(ntV, 0, n);
        DAAL_CHECK_BLOCK_STATUS(vBlock);
        algorithmFPType * tV = vBlock.get();
        for (size_t i = 0; i < nComponents; i++)
        {
            PRAGMA_IVDEP
            for (size_t j = 0; j < n; j++)
            {
                tV[i * n + j] = V[i * n + j];
            }
        }
    }

    return (ec ? Status(ec) : Status());
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
