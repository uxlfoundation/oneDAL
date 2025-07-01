/* file: service_blas.h */
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
//  Template wrappers for BLAS functions.
//--
*/

#ifndef __SERVICE_BLAS_H__
#define __SERVICE_BLAS_H__

#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"

#include "src/externals/config.h"

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
class Helper
{
public:
    static inline void copy(fpType * dsc, const fpType * src, const size_t n);
};

template <typename fpType, CpuType cpu>
inline void Helper<fpType, cpu>::copy(fpType * dsc, const fpType * src, const size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        dsc[i] = src[i];
    }
}

#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)

template <>
inline void Helper<float, avx512>::copy(float * dsc, const float * src, const size_t n)
{
    size_t i           = 0;
    const size_t align = ((64 - (reinterpret_cast<size_t>(dsc) & 63)) & 63) >> 2;
    for (; i < align; ++i)
    {
        dsc[i] = src[i];
    }
    for (; (i + 16) < n; i += 16)
    {
        const __m512 srcVec = _mm512_loadu_ps(&src[i]);
        _mm512_stream_ps(&dsc[i], srcVec);
    }
    for (; i < n; i++)
    {
        dsc[i] = src[i];
    }
}

template <>
inline void Helper<double, avx512>::copy(double * dsc, const double * src, const size_t n)
{
    size_t i           = 0;
    const size_t align = ((64 - (reinterpret_cast<size_t>(dsc) & 63)) & 63) >> 3;
    for (; i < align; ++i)
    {
        dsc[i] = src[i];
    }
    for (; (i + 8) < n; i += 8)
    {
        const __m512d srcVec = _mm512_loadu_pd(&src[i]);
        _mm512_stream_pd(&dsc[i], srcVec);
    }
    for (; i < n; i++)
    {
        dsc[i] = src[i];
    }
}

#endif

/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl>
struct Blas
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;

    static void xsyrk(const char * uplo, const char * trans, const SizeType * p, const SizeType * n, const fpType * alpha, const fpType * a,
                      const SizeType * lda, const fpType * beta, fpType * ata, const SizeType * ldata)
    {
        _impl<fpType, cpu>::xsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xxsyrk(const char * uplo, const char * trans, const SizeType * p, const SizeType * n, const fpType * alpha, const fpType * a,
                       const SizeType * lda, const fpType * beta, fpType * ata, const SizeType * ldata)
    {
        _impl<fpType, cpu>::xxsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xsyr(const char * uplo, const SizeType * n, const fpType * alpha, const fpType * x, const SizeType * incx, fpType * a,
                     const SizeType * lda)
    {
        _impl<fpType, cpu>::xsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xxsyr(const char * uplo, const SizeType * n, const fpType * alpha, const fpType * x, const SizeType * incx, fpType * a,
                      const SizeType * lda)
    {
        _impl<fpType, cpu>::xxsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xgemm(const char * transa, const char * transb, const SizeType * p, const SizeType * ny, const SizeType * n, const fpType * alpha,
                      const fpType * a, const SizeType * lda, const fpType * y, const SizeType * ldy, const fpType * beta, fpType * aty,
                      const SizeType * ldaty)
    {
        _impl<fpType, cpu>::xgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xxgemm(const char * transa, const char * transb, const SizeType * p, const SizeType * ny, const SizeType * n, const fpType * alpha,
                       const fpType * a, const SizeType * lda, const fpType * y, const SizeType * ldy, const fpType * beta, fpType * aty,
                       const SizeType * ldaty)
    {
        _impl<fpType, cpu>::xxgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xsymm(const char * side, const char * uplo, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a,
                      const SizeType * lda, const fpType * b, const SizeType * ldb, const fpType * beta, fpType * c, const SizeType * ldc)
    {
        _impl<fpType, cpu>::xsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xxsymm(char * side, char * uplo, SizeType * m, SizeType * n, fpType * alpha, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                       fpType * beta, fpType * c, SizeType * ldc)
    {
        _impl<fpType, cpu>::xxsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xgemv(const char * trans, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a, const SizeType * lda,
                      const fpType * x, const SizeType * incx, const fpType * beta, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xxgemv(const char * trans, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a, const SizeType * lda,
                       const fpType * x, const SizeType * incx, const fpType * beta, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xxgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xaxpy(SizeType * n, fpType * a, fpType * x, SizeType * incx, fpType * y, SizeType * incy)
    {
        _impl<fpType, cpu>::xaxpy(n, a, x, incx, y, incy);
    }

    static void xxaxpy(const SizeType * n, const fpType * a, const fpType * x, const SizeType * incx, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xxaxpy(n, a, x, incx, y, incy);
    }

    static fpType xxdot(const SizeType * n, const fpType * x, const SizeType * incx, const fpType * y, const SizeType * incy)
    {
        return _impl<fpType, cpu>::xxdot(n, x, incx, y, incy);
    }

    static void xscal(const SizeType * n, const fpType * a, fpType * x, const SizeType * incx) { _impl<fpType, cpu>::xscal(n, a, x, incx); }

    static services::Status xgemm_blocked(const char * transa, const char * transb, const SizeType * na, const SizeType * nb, const SizeType * cols,
                                          const fpType * alpha, const NumericTable * ta, const SizeType * lda, const NumericTable * tb,
                                          const SizeType * ldb, const fpType * beta, NumericTable * tc, const SizeType * ldc, int reqBlockSizeA = -1,
                                          int reqBlockSizeB = -1)
    {
        SafeStatus safeStat;

        SizeType nRowsA = *na;
        SizeType nRowsB = *nb;
        SizeType nCols  = *cols;

        /* Read block sizes from parameters or set to deafault value */
        int blockSizeA = (reqBlockSizeA > 0) ? reqBlockSizeA : 128;
        int blockSizeB = (reqBlockSizeB > 0) ? reqBlockSizeB : 128;

        /* Block size cannot be greater than whole number of rows */
        blockSizeA = (blockSizeA > nRowsA) ? nRowsA : blockSizeA;
        blockSizeB = (blockSizeB > nRowsB) ? nRowsB : blockSizeB;

        /* Number of blocks */
        SizeType nBlocksA = nRowsA / blockSizeA;
        SizeType nBlocksB = nRowsB / blockSizeB;

        /* Last block size */
        SizeType lastBlockSizeA = nRowsA - nBlocksA * blockSizeA;
        SizeType lastBlockSizeB = nRowsB - nBlocksB * blockSizeB;

        /* Increase the number of blocks if last block size is nonzero */
        if (lastBlockSizeA != 0)
        {
            nBlocksA++;
        }
        else
        {
            lastBlockSizeA = blockSizeA;
        }
        if (lastBlockSizeB != 0)
        {
            nBlocksB++;
        }
        else
        {
            lastBlockSizeB = blockSizeB;
        }

        const bool isSOARes = tc->getDataLayout() & NumericTableIface::soa;
        TlsMem<fpType, cpu> tlsMklBuff(blockSizeA * blockSizeB);

        /* Threaded loop by whole number of blocks */
        daal::conditional_threader_for((nRowsB > blockSizeB * 2), nBlocksB, [&, isSOARes](SizeType iBlockB) {
            /* Current block size - can be less than general block size for last block */
            SizeType nRowsInBlockB   = (iBlockB < (nBlocksB - 1)) ? blockSizeB : lastBlockSizeB;
            const SizeType startRowB = iBlockB * blockSizeB;

            ReadRows<fpType, cpu> mtb(*const_cast<NumericTable *>(tb), startRowB, nRowsInBlockB);
            DAAL_CHECK_BLOCK_STATUS_THR(mtb);
            const fpType * const b = mtb.get();

            WriteOnlyRows<fpType, cpu> mtcRows;
            if (!isSOARes)
            {
                /* Get pointer to write resulted rows */
                mtcRows.set(tc, startRowB, nRowsInBlockB);
                DAAL_CHECK_MALLOC_THR(mtcRows.get());
            }

            daal::conditional_threader_for((nRowsA > blockSizeA * 2), nBlocksA, [&](SizeType iBlockA) {
                /* Current block size - can be less than general block size for last block */
                SizeType nRowsInBlockA   = (iBlockA < (nBlocksA - 1)) ? blockSizeA : lastBlockSizeA;
                const SizeType startRowA = iBlockA * blockSizeA;

                /* Read rows for numeric tables */
                ReadRows<fpType, cpu> mta(*const_cast<NumericTable *>(ta), startRowA, nRowsInBlockA);
                DAAL_CHECK_BLOCK_STATUS_THR(mta);
                const fpType * const a = mta.get();

                /* Call to sequential GEMM */
                if (!isSOARes)
                {
                    fpType * const c = mtcRows.get() + startRowA;
                    xxgemm(transa, transb, &nRowsInBlockA, &nRowsInBlockB, &nCols, alpha, a, lda, b, ldb, beta, c, ldc);
                }
                else
                {
                    fpType * const c = tlsMklBuff.local();
                    DAAL_CHECK_MALLOC_THR(c);
                    DAAL_INT ldc2 = blockSizeB;

                    xxgemm(transa, transb, &nRowsInBlockB, &nRowsInBlockA, &nCols, alpha, b, ldb, a, lda, beta, c, &ldc2);
                    for (size_t i = 0; i < nRowsInBlockA; ++i)
                    {
                        WriteOnlyColumns<fpType, cpu> mtcColumns(tc, startRowA + i, startRowB, nRowsInBlockB);
                        DAAL_CHECK_BLOCK_STATUS_THR(mtcColumns);
                        Helper<fpType, cpu>::copy(mtcColumns.get(), c + i * ldc2, nRowsInBlockB);
                    }
                }
            });
        });

        return safeStat.detach();
    } /* xgemm_blocked */
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using BlasInst = Blas<fpType, cpu, BlasBackend>;
} // namespace internal
} // namespace daal

#endif
