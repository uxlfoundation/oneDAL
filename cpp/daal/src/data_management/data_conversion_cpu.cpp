/** file data_conversion_cpu.cpp */
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

#include "src/data_management/data_conversion_cpu.h"
#include "data_management/data/internal/conversion.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_dispatch.h"

namespace daal
{
namespace data_management
{
namespace internal
{
/* only for AVX512 architecture with using intrinsics */
#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)
template <typename T>
void vectorCopyInternal()
{
    vectorCopyInternal<T>();
}

template <>
void vectorCopyInternal<float>(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    float * pd           = static_cast<float *>(dst);
    float const * pmin   = static_cast<float const *>(ptrMin);
    char const * ptrByte = static_cast<char const *>(ptrMin);

    const size_t nColSize = ncols - ncols % 8;

    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < nColSize; j += 8)
        {
            __m256 ps = _mm512_i64gather_ps(*((__m512i *)&arrOffsets[j]), pmin + i, 1);
            _mm256_storeu_ps(pd + i * ncols + j, ps);
        }
        for (size_t j = nColSize; j < ncols; ++j)
        {
            char const * pc   = ptrByte + arrOffsets[j];
            pd[i * ncols + j] = *((float *)(pc) + i);
        }
    }
}

template <>
void vectorCopyInternal<double>(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    double * pd          = static_cast<double *>(dst);
    double const * pmin  = static_cast<double const *>(ptrMin);
    char const * ptrByte = static_cast<char const *>(ptrMin);

    const size_t nColSize = ncols - ncols % 8;

    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < nColSize; j += 8)
        {
            __m512d p = _mm512_i64gather_pd(*((__m512i *)&arrOffsets[j]), pmin + i, 1);
            _mm512_storeu_pd(pd + i * ncols + j, p);
        }
        for (size_t j = nColSize; j < ncols; ++j)
        {
            char const * pc   = ptrByte + arrOffsets[j];
            pd[i * ncols + j] = *((double *)(pc) + i);
        }
    }
}

template <typename T, CpuType cpu>
void vectorCopy(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    vectorCopy<T, cpu>(nrows, ncols, dst, ptrMin, arrOffsets);
}

/* Convert float to float from columnar to row major format using AVX512 architecture */
template <>
void vectorCopy<float, avx512>(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    vectorCopyInternal<float>(nrows, ncols, dst, ptrMin, arrOffsets);
}

/* Convert double to double from columnar to row major format using AVX512 architecture */
template <>
void vectorCopy<double, avx512>(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    vectorCopyInternal<double>(nrows, ncols, dst, ptrMin, arrOffsets);
}
#endif

template <typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, const void * src, void * dst)
{
    for (size_t i = 0; i < n; i++)
    {
        ((T2 *)dst)[i] = static_cast<T2>(((T1 *)src)[i]);
    }
}

template <typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride)
{
    for (size_t i = 0; i < n; i++)
    {
        *(T2 *)(((char *)dst) + i * dstByteStride) = static_cast<T2>(*(T1 *)(((char *)src) + i * srcByteStride));
    }
}

#undef DAAL_FUNCS_UP_ENTRY
#define DAAL_FUNCS_UP_ENTRY(F, T, A)        \
    template void F<T, float, DAAL_CPU> A;  \
    template void F<T, double, DAAL_CPU> A; \
    template void F<T, int, DAAL_CPU> A;

#undef DAAL_FUNCS_DOWN_ENTRY
#define DAAL_FUNCS_DOWN_ENTRY(F, T, A)      \
    template void F<float, T, DAAL_CPU> A;  \
    template void F<double, T, DAAL_CPU> A; \
    template void F<int, T, DAAL_CPU> A;

#undef DAAL_CONVERT_UP_FUNCS
#define DAAL_CONVERT_UP_FUNCS(F, A)          \
    DAAL_FUNCS_UP_ENTRY(F, float, A)         \
    DAAL_FUNCS_UP_ENTRY(F, double, A)        \
    DAAL_FUNCS_UP_ENTRY(F, int, A)           \
    DAAL_FUNCS_UP_ENTRY(F, unsigned int, A)  \
    DAAL_FUNCS_UP_ENTRY(F, DAAL_INT64, A)    \
    DAAL_FUNCS_UP_ENTRY(F, DAAL_UINT64, A)   \
    DAAL_FUNCS_UP_ENTRY(F, char, A)          \
    DAAL_FUNCS_UP_ENTRY(F, unsigned char, A) \
    DAAL_FUNCS_UP_ENTRY(F, short, A)         \
    DAAL_FUNCS_UP_ENTRY(F, unsigned short, A)

#undef DAAL_CONVERT_DOWN_FUNCS
#define DAAL_CONVERT_DOWN_FUNCS(F, A)          \
    DAAL_FUNCS_DOWN_ENTRY(F, unsigned int, A)  \
    DAAL_FUNCS_DOWN_ENTRY(F, DAAL_INT64, A)    \
    DAAL_FUNCS_DOWN_ENTRY(F, DAAL_UINT64, A)   \
    DAAL_FUNCS_DOWN_ENTRY(F, char, A)          \
    DAAL_FUNCS_DOWN_ENTRY(F, unsigned char, A) \
    DAAL_FUNCS_DOWN_ENTRY(F, short, A)         \
    DAAL_FUNCS_DOWN_ENTRY(F, unsigned short, A)

DAAL_CONVERT_UP_FUNCS(vectorConvertFuncCpu, (size_t n, const void * src, void * dst))
DAAL_CONVERT_DOWN_FUNCS(vectorConvertFuncCpu, (size_t n, const void * src, void * dst))

DAAL_CONVERT_UP_FUNCS(vectorStrideConvertFuncCpu, (size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride))
DAAL_CONVERT_DOWN_FUNCS(vectorStrideConvertFuncCpu, (size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride))

template <typename T, CpuType cpu>
void vectorAssignValueToArrayCpu(void * const ptr, const size_t n, const void * const value)
{
    const T & valueT = *((const T *)value);
    T * const ptrT   = (T *)ptr;
    services::internal::service_memset<T, cpu>(ptrT, valueT, n);
}

#define DAAL_REGISTER_VECTOR_ASSIGN_CPU(Type) \
    template void vectorAssignValueToArrayCpu<Type, DAAL_CPU>(void * const ptr, const size_t n, const void * const value);
DAAL_REGISTER_WITH_HOMOGEN_NT_TYPES(DAAL_REGISTER_VECTOR_ASSIGN_CPU)

} // namespace internal
} // namespace data_management
} // namespace daal
