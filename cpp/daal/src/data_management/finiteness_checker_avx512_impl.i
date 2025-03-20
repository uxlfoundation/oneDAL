/** file: finiteness_checker_avx512_impl.i */
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

#ifndef __FINITENESS_CHECKER_AVX512_IMPL_I__
#define __FINITENESS_CHECKER_AVX512_IMPL_I__

#ifndef ONEDAL_XSIMD_ENABLED

/*
// Computes sum of the elements of input array of type `float` with AVX512 instructions.
*/
template <>
float sumWithSIMD<float, avx512>(size_t n, const float * dataPtr)
{
    constexpr size_t avx512RegisterLength = 512;
    constexpr size_t numberOfBitsInByte   = 8;
    constexpr size_t nPerInstr            = avx512RegisterLength / (numberOfBitsInByte * sizeof(float));
    float sum;

    __m512 sums     = _mm512_set1_ps(0);
    __m512 * ptr512 = (__m512 *)dataPtr;
    for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm512_add_ps(sums, ptr512[i]);
    sum = _mm512_reduce_add_ps(sums);

    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

/*
// Computes sum of the elements of input array of type `double` with AVX512 instructions.
*/
template <>
double sumWithSIMD<double, avx512>(size_t n, const double * dataPtr)
{
    constexpr size_t avx512RegisterLength = 512;
    constexpr size_t numberOfBitsInByte   = 8;
    constexpr size_t nPerInstr            = avx512RegisterLength / (numberOfBitsInByte * sizeof(double));
    double sum;

    __m512d sums     = _mm512_set1_pd(0);
    __m512d * ptr512 = (__m512d *)dataPtr;
    for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm512_add_pd(sums, ptr512[i]);
    sum = _mm512_reduce_add_pd(sums);

    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

#endif // ONEDAL_XSIMD_ENABLED

template <>
float computeSum<float, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<float, avx512>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx512>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<double, avx512>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSumSOA<avx512>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    // computeSumSOASIMD defined in finiteness_checker_cpu.cpp
    return computeSumSOASIMD<avx512>(table, sumIsFinite, st);
}

#ifndef ONEDAL_XSIMD_ENABLED

/*
// Computes finiteness for float data using multi-threading and AVX512 instructions which match isnan and isinf
*/
template <>
services::Status checkFinitenessInBlocks<avx512>(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr,
                                                 size_t nPerBlock, size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    constexpr size_t avx512RegisterLength = 512;
    constexpr size_t numberOfBitsInByte   = 8;
    constexpr size_t nPerInstr            = avx512RegisterLength / (numberOfBitsInByte * sizeof(float));
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
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
        __m512i exp512Mask  = _mm512_set1_epi32(floatExpMask);
        __m512i frac512Mask = _mm512_set1_epi32(floatFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask16 notAllowNaNMask =
            allowNaN ? _cvtu32_mask16(0) : _cvtu32_mask16(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);
        __mmask16 endMask = _cvtu32_mask16(0);

        __m512i * ptr512i = (__m512i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m512i expBits  = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i fracBits = _mm512_and_si512(frac512Mask, ptr512i[i]);

            __mmask16 expAreOnes   = _mm512_cmpeq_epi32_mask(exp512Mask, expBits);
            __mmask16 fracAreZeros = _mm512_cmpeq_epi32_mask(zero512, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask16 orMask    = _kor_mask16(fracAreZeros, notAllowNaNMask);
            __mmask16 finalMask = _kand_mask16(expAreOnes, orMask);
            endMask             = _kor_mask16(endMask, finalMask);
        }
        if (_cvtmask16_u32(endMask) != 0) notFinitePtr[iBlock] = true;

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
// Computes finiteness for double data using multi-threading and AVX512 instructions which match isnan and isinf
*/
template <>
services::Status checkFinitenessInBlocks<avx512>(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr,
                                                 size_t nPerBlock, size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    constexpr size_t avx512RegisterLength = 512;
    constexpr size_t numberOfBitsInByte   = 8;
    constexpr size_t nPerInstr            = avx512RegisterLength / (numberOfBitsInByte * sizeof(double));
    services::internal::TArray<bool, avx512> notFiniteArr(nTotalBlocks);
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
        __m512i exp512Mask  = _mm512_set1_epi64(doubleExpMask);
        __m512i frac512Mask = _mm512_set1_epi64(doubleFracMask);
        __m512i zero512     = _mm512_setzero_si512();

        __mmask8 notAllowNaNMask =
            allowNaN ? _cvtu32_mask8(0) : _cvtu32_mask8(static_cast<unsigned int>(services::internal::MaxVal<int>::get()) * 2 + 1);
        __mmask8 endMask = _cvtu32_mask8(0);

        __m512i * ptr512i = (__m512i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m512i expBits  = _mm512_and_si512(exp512Mask, ptr512i[i]);
            __m512i fracBits = _mm512_and_si512(frac512Mask, ptr512i[i]);

            __mmask8 expAreOnes   = _mm512_cmpeq_epi64_mask(exp512Mask, expBits);
            __mmask8 fracAreZeros = _mm512_cmpeq_epi64_mask(zero512, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __mmask8 orMask    = _kor_mask8(fracAreZeros, notAllowNaNMask);
            __mmask8 finalMask = _kand_mask8(expAreOnes, orMask);
            endMask            = _kor_mask8(endMask, finalMask);
        }
        if (_cvtmask8_u32(endMask) != 0) notFinitePtr[iBlock] = true;

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
bool checkFiniteness<float, avx512>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs, bool allowNaN)
{
    // checkFinitenessSIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSIMD<float, avx512>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx512>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs, bool allowNaN)
{
    // checkFinitenessSIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSIMD<double, avx512>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFinitenessSOA<avx512>(NumericTable & table, bool allowNaN, services::Status & st)
{
    // checkFinitenessSOASIMD defined in finiteness_checker_cpu.cpp
    return checkFinitenessSOASIMD<avx512>(table, allowNaN, st);
}

#endif // __FINITENESS_CHECKER_AVX512_IMPL_I__
