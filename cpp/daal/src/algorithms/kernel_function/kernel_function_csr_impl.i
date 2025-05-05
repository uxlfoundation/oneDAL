/* file: kernel_function_csr_impl.i */
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
//  Common kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_CSR_IMPL_I__
#define __KERNEL_FUNCTION_CSR_IMPL_I__

#if defined(DAAL_INTEL_CPP_COMPILER)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
algorithmFPType computeDotProductBaseline(const size_t startIndexA, const size_t endIndexA, const algorithmFPType * valuesA, const size_t * indicesA,
                                          const size_t startIndexB, const size_t endIndexB, const algorithmFPType * valuesB, const size_t * indicesB)
{
    size_t offsetA      = startIndexA;
    size_t offsetB      = startIndexB;
    algorithmFPType sum = 0.0;
    while ((offsetA < endIndexA) && (offsetB < endIndexB))
    {
        size_t colIndex1 = indicesA[offsetA];
        size_t colIndex2 = indicesB[offsetB];
        if (colIndex1 == colIndex2)
        {
            sum += valuesA[offsetA] * valuesB[offsetB];
            offsetA++;
            offsetB++;
        }
        else if (colIndex1 > colIndex2)
        {
            offsetB++;
        }
        else // (colIndex1 < colIndex2)
        {
            offsetA++;
        }
    }
    return sum;
}

template <typename algorithmFPType, CpuType cpu>
algorithmFPType KernelCSRImplBase<algorithmFPType, cpu>::computeDotProduct(const size_t startIndexA, const size_t endIndexA,
                                                                           const algorithmFPType * valuesA, const size_t * indicesA,
                                                                           const size_t startIndexB, const size_t endIndexB,
                                                                           const algorithmFPType * valuesB, const size_t * indicesB)
{
    return computeDotProductBaseline<algorithmFPType, cpu>(startIndexA, endIndexA, valuesA, indicesA, startIndexB, endIndexB, valuesB, indicesB);
}

#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)

    #undef __DAAL_IA32e

    #if defined(_M_AMD64) || defined(__amd64) || defined(__x86_64) || defined(__x86_64__)
        #define __DAAL_IA32e
    #endif

    #if defined(__DAAL_IA32e)

        #if (__CPUID__(DAAL_CPU) == __avx512__)

template <>
inline double KernelCSRImplBase<double, avx512>::computeDotProduct(const size_t startIndexA, const size_t endIndexA, const double * valuesA,
                                                                   const size_t * indicesA, const size_t startIndexB, const size_t endIndexB,
                                                                   const double * valuesB, const size_t * indicesB)
{
    size_t offsetA = startIndexA;
    size_t offsetB = startIndexB;
    double sum     = 0.0;

    if (offsetA + 8 <= endIndexA && offsetB + 8 <= endIndexB && !(endIndexA & 0xffffffff00000000) && !(endIndexB & 0xffffffff00000000))
    {
        const __m512i all_31 = _mm512_set1_epi32(31);
        const __m512i all_f  = _mm512_set1_epi32(0xffffffff);
        const __m512i upcon  = _mm512_set_epi32(0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0);
        const __m512i idx    = _mm512_set_epi32(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);

        __m512i tmpA = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
        __m512i tmpB = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
        __m512i idxA = _mm512_permutexvar_epi32(idx, tmpA);
        __m512i idxB = _mm512_permutexvar_epi32(idx, tmpB);

        /* Block of 8 indices */
        __m256i iA = _mm512_extracti64x4_epi64(idxA, 0);
        __m256i iB = _mm512_extracti64x4_epi64(idxB, 0);
        /* Block of 8 values */
        __m512d valA = _mm512_loadu_pd(valuesA + offsetA);
        __m512d valB = _mm512_loadu_pd(valuesB + offsetB);
        __m512d vSum = _mm512_setzero_pd();

        while (1)
        {
            __m512i concatenated = _mm512_inserti64x4(_mm512_castsi256_si512(iA), iB, 1);                                   // put b above a
            __m512i matches    = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(_mm512_conflict_epi32(concatenated), 1)); // take top half of result
            __m512i perm_idx   = _mm512_sub_epi32(all_31, _mm512_lzcnt_epi32(matches));
            __m512i perm_idx64 = _mm512_mask_permutevar_epi32(_mm512_setzero_epi32(), 0x5555, upcon, perm_idx);
            __mmask8 have_match  = (__mmask8)_mm512_test_epi32_mask(matches, all_f);
            __m512d matched_vals = _mm512_maskz_permutexvar_pd(have_match, perm_idx64, valA);
            vSum                 = _mm512_fmadd_pd(matched_vals, valB, vSum);

            const int * aidx = (const int *)(&iA);
            const int * bidx = (const int *)(&iB);
            int a7           = aidx[7];
            int b7           = bidx[7];
            if (a7 == b7)
            {
                offsetA += 8;
                offsetB += 8;
                if (offsetA + 8 > endIndexA || offsetB + 8 > endIndexB)
                {
                    break;
                }
                tmpA = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
                tmpB = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
                idxA = _mm512_permutexvar_epi32(idx, tmpA);
                idxB = _mm512_permutexvar_epi32(idx, tmpB);
                iA   = _mm512_extracti64x4_epi64(idxA, 0);
                iB   = _mm512_extracti64x4_epi64(idxB, 0);

                valA = _mm512_loadu_pd(valuesA + offsetA);
                valB = _mm512_loadu_pd(valuesB + offsetB);
            }
            else if (a7 > b7)
            {
                offsetB += 8;
                if (offsetB + 8 > endIndexB)
                {
                    break;
                }
                tmpB = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
                idxB = _mm512_permutexvar_epi32(idx, tmpB);
                iB   = _mm512_extracti64x4_epi64(idxB, 0);

                valB = _mm512_loadu_pd(valuesB + offsetB);
            }
            else // (a7 < b7)
            {
                offsetA += 8;
                if (offsetA + 8 > endIndexA)
                {
                    break;
                }
                tmpA = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
                idxA = _mm512_permutexvar_epi32(idx, tmpA);
                iA   = _mm512_extracti64x4_epi64(idxA, 0);

                valA = _mm512_loadu_pd(valuesA + offsetA);
            }
        }

        double partialSum[8];
        _mm512_storeu_pd(partialSum, vSum);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (int i = 0; i < 8; i++)
        {
            sum += partialSum[i];
        }
    }

    /* Process tail elements in scalar loop */
    sum += computeDotProductBaseline<double, avx512>(offsetA, endIndexA, valuesA, indicesA, offsetB, endIndexB, valuesB, indicesB);

    return sum;
}

template <>
inline float KernelCSRImplBase<float, avx512>::computeDotProduct(const size_t startIndexA, const size_t endIndexA, const float * valuesA,
                                                                 const size_t * indicesA, const size_t startIndexB, const size_t endIndexB,
                                                                 const float * valuesB, const size_t * indicesB)
{
    size_t offsetA = startIndexA;
    size_t offsetB = startIndexB;
    double sum     = 0.0;

    if (offsetA + 8 <= endIndexA && offsetB + 8 <= endIndexB && !(endIndexA & 0xffffffff00000000) && !(endIndexB & 0xffffffff00000000))
    {
        const __m512i all_31 = _mm512_set1_epi32(31);
        const __m512i all_f  = _mm512_set1_epi32(0xffffffff);
        const __m512i upcon  = _mm512_set_epi32(0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0);
        const __m512i idx    = _mm512_set_epi32(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
        __m512i tmpA         = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
        __m512i tmpB         = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
        __m512i idxA         = _mm512_permutexvar_epi32(idx, tmpA);
        __m512i idxB         = _mm512_permutexvar_epi32(idx, tmpB);

        /* Block of 8 indices */
        __m256i iA = _mm512_extracti64x4_epi64(idxA, 0);
        __m256i iB = _mm512_extracti64x4_epi64(idxB, 0);
        /* Block of 8 values */
        __m256 valFloatA = _mm256_loadu_ps(valuesA + offsetA);
        __m256 valFloatB = _mm256_loadu_ps(valuesB + offsetB);
        __m512d valA     = _mm512_cvt_roundps_pd(valFloatA, _MM_FROUND_NO_EXC);
        __m512d valB     = _mm512_cvt_roundps_pd(valFloatB, _MM_FROUND_NO_EXC);
        __m512d vSum     = _mm512_setzero_pd();

        while (1)
        {
            __m512i concatenated = _mm512_inserti64x4(_mm512_castsi256_si512(iA), iB, 1);                                   // put b above a
            __m512i matches    = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(_mm512_conflict_epi32(concatenated), 1)); // take top half of result
            __m512i perm_idx   = _mm512_sub_epi32(all_31, _mm512_lzcnt_epi32(matches));
            __m512i perm_idx64 = _mm512_mask_permutevar_epi32(_mm512_setzero_epi32(), 0x5555, upcon, perm_idx);
            __mmask8 have_match  = (__mmask8)_mm512_test_epi32_mask(matches, all_f);
            __m512d matched_vals = _mm512_maskz_permutexvar_pd(have_match, perm_idx64, valA);
            vSum                 = _mm512_fmadd_pd(matched_vals, valB, vSum);

            const int * aidx = (const int *)(&iA);
            const int * bidx = (const int *)(&iB);
            int a7           = aidx[7];
            int b7           = bidx[7];
            if (a7 == b7)
            {
                offsetA += 8;
                offsetB += 8;
                if (offsetA + 8 > endIndexA || offsetB + 8 > endIndexB)
                {
                    break;
                }
                tmpA = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
                tmpB = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
                idxA = _mm512_permutexvar_epi32(idx, tmpA);
                idxB = _mm512_permutexvar_epi32(idx, tmpB);
                iA   = _mm512_extracti64x4_epi64(idxA, 0);
                iB   = _mm512_extracti64x4_epi64(idxB, 0);

                valFloatA = _mm256_loadu_ps(valuesA + offsetA);
                valFloatB = _mm256_loadu_ps(valuesB + offsetB);
                valA      = _mm512_cvt_roundps_pd(valFloatA, _MM_FROUND_NO_EXC);
                valB      = _mm512_cvt_roundps_pd(valFloatB, _MM_FROUND_NO_EXC);
            }
            else if (a7 > b7)
            {
                offsetB += 8;
                if (offsetB + 8 > endIndexB)
                {
                    break;
                }
                tmpB = _mm512_loadu_si512((__m512i *)(indicesB + offsetB));
                idxB = _mm512_permutexvar_epi32(idx, tmpB);
                iB   = _mm512_extracti64x4_epi64(idxB, 0);

                valFloatB = _mm256_loadu_ps(valuesB + offsetB);
                valB      = _mm512_cvt_roundps_pd(valFloatB, _MM_FROUND_NO_EXC);
            }
            else // (a7 < b7)
            {
                offsetA += 8;
                if (offsetA + 8 > endIndexA)
                {
                    break;
                }
                tmpA = _mm512_loadu_si512((__m512i *)(indicesA + offsetA));
                idxA = _mm512_permutexvar_epi32(idx, tmpA);
                iA   = _mm512_extracti64x4_epi64(idxA, 0);

                valFloatA = _mm256_loadu_ps(valuesA + offsetA);
                valA      = _mm512_cvt_roundps_pd(valFloatA, _MM_FROUND_NO_EXC);
            }
        }

        double partialSum[8];
        _mm512_storeu_pd(partialSum, vSum);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (int i = 0; i < 8; i++)
        {
            sum += partialSum[i];
        }
    }

    /* Process tail elements in scalar loop */
    sum += computeDotProductBaseline<float, avx512>(offsetA, endIndexA, valuesA, indicesA, offsetB, endIndexB, valuesB, indicesB);

    return (float)sum;
}
        #endif // __CPUID__(DAAL_CPU) == __avx512__

    #endif // __DAAL_IA32e
#endif     // DAAL_INTEL_CPP_COMPILER
} // namespace internal
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
