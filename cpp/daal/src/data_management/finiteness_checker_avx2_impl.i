/** file: finiteness_checker_avx2_impl.i */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#ifndef __FINITENESS_CHECKER_AVX2_IMPL_I__
#define __FINITENESS_CHECKER_AVX2_IMPL_I__

#ifndef ONEDAL_XSIMD_ENABLED

/*
// Computes sum of the elements of input array of type `float` with AVX2 instructions.
*/
template <>
float sumWithSIMD<float, avx2>(size_t n, const float * dataPtr)
{
    constexpr size_t avx2RegisterLength = 256;
    constexpr size_t numberOfBitsInByte = 8;
    constexpr size_t nPerInstr          = avx2RegisterLength / (numberOfBitsInByte * sizeof(float));
    float sum;

    __m256 sums     = _mm256_set1_ps(0);
    __m256 * ptr256 = (__m256 *)dataPtr;
    for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_ps(sums, ptr256[i]);

    // AVX2 doesn't have reduce_add_ps, so finer-grained used of intrinsics necessary
    __m128 s0    = _mm256_castps256_ps128(sums);
    __m128 s1    = _mm256_extractf128_ps(sums, 1);
    s0           = _mm_add_ps(s0, s1);
    __m128 stemp = _mm_movehl_ps(s0, s0);
    __m128 sD    = _mm_add_ps(s0, stemp);
    __m128 hi    = _mm_shuffle_ps(sD, sD, 0x1);

    sum = _mm_cvtss_f32(_mm_add_ss(sD, hi));

    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

/*
// Computes sum of the elements of input array of type `double` with AVX2 instructions.
*/
template <>
double sumWithSIMD<double, avx2>(size_t n, const double * dataPtr)
{
    constexpr size_t avx2RegisterLength = 256;
    constexpr size_t numberOfBitsInByte = 8;
    constexpr size_t nPerInstr          = avx2RegisterLength / (numberOfBitsInByte * sizeof(double));
    double sum;

    __m256d sums     = _mm256_set1_pd(0);
    __m256d * ptr256 = (__m256d *)dataPtr;
    for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_pd(sums, ptr256[i]);

    // AVX2 doesn't have reduce_add_pd, so finer-grained used of intrinsics necessary
    __m128d s0    = _mm256_castpd256_pd128(sums);
    __m128d s1    = _mm256_extractf128_pd(sums, 1);
    s0            = _mm_add_pd(s0, s1);
    __m128d stemp = _mm_unpackhi_pd(s0, s0);

    sum = _mm_cvtsd_f64(_mm_add_sd(s0, stemp));

    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

#endif // ONEDAL_XSIMD_ENABLED

template <>
float computeSum<float, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<float, avx2>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<double, avx2>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSumSOA<avx2>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    // computeSumSOASIMD defined in finiteness_checker_cpu.cpp
    return computeSumSOASIMD<avx2>(table, sumIsFinite, st);
}

#ifndef ONEDAL_XSIMD_ENABLED
/*
// Computes finiteness for float data using multi-threading and AVX2 instructions which match isnan and isinf
*/
template <>
services::Status checkFinitenessInBlocks<avx2>(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                               size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    constexpr size_t avx2RegisterLength = 256;
    constexpr size_t numberOfBitsInByte = 8;
    constexpr size_t nPerInstr          = avx2RegisterLength / (numberOfBitsInByte * sizeof(float));
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi32(floatExpMask);
        __m256i frac256Mask = _mm256_set1_epi32(floatFracMask);
        __m256i zero256     = _mm256_setzero_si256();
        __m256i endMask     = _mm256_setzero_si256();

        __m256i notAllowNaNMask = allowNaN ? _mm256_setzero_si256() : _mm256_set1_epi64x(-1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __m256i expAreOnes   = _mm256_cmpeq_epi32(exp256Mask, expBits);
            __m256i fracAreZeros = _mm256_cmpeq_epi32(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __m256i orMask    = _mm256_or_si256(fracAreZeros, notAllowNaNMask);
            __m256i finalMask = _mm256_and_si256(expAreOnes, orMask);
            endMask           = _mm256_or_si256(endMask, finalMask); // collect ones for final check
        }
        if (_mm256_testz_si256(endMask, endMask) != 1) notFinitePtr[iBlock] = true;
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

/*
// Computes finiteness for double data using multi-threading and AVX2 instructions which match isnan and isinf
*/
template <>
services::Status checkFinitenessInBlocks<avx2>(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                               size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    constexpr size_t avx2RegisterLength = 256;
    constexpr size_t numberOfBitsInByte = 8;
    constexpr size_t nPerInstr          = avx2RegisterLength / (numberOfBitsInByte * sizeof(double));
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi64x(doubleExpMask);
        __m256i frac256Mask = _mm256_set1_epi64x(doubleFracMask);
        __m256i zero256     = _mm256_setzero_si256();
        __m256i endMask     = _mm256_setzero_si256();

        __m256i notAllowNaNMask = allowNaN ? _mm256_setzero_si256() : _mm256_set1_epi64x(-1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __m256i expAreOnes   = _mm256_cmpeq_epi64(exp256Mask, expBits);
            __m256i fracAreZeros = _mm256_cmpeq_epi64(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __m256i orMask    = _mm256_or_si256(fracAreZeros, notAllowNaNMask);
            __m256i finalMask = _mm256_and_si256(expAreOnes, orMask);
            endMask           = _mm256_or_si256(endMask, finalMask); // collect ones for final check
        }
        if (_mm256_testz_si256(endMask, endMask) != 1) notFinitePtr[iBlock] = true;

        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

#endif // #ifndef ONEDAL_XSIMD_ENABLED

template <>
bool checkFiniteness<float, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs, bool allowNaN)
{
    // checkFinitenessSIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSIMD<float, avx2>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs, bool allowNaN)
{
    // checkFinitenessSIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSIMD<double, avx2>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFinitenessSOA<avx2>(NumericTable & table, bool allowNaN, services::Status & st)
{
    // checkFinitenessSOASIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSOASIMD<avx2>(table, allowNaN, st);
}

#endif // __FINITENESS_CHECKER_AVX2_IMPL_I__
