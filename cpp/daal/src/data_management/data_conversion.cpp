/** file data_management_utils.cpp */
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
//  Implementation for data conversions dispatcher
//--
*/

#include "services/internal/daal_kernel_defines.h"
#include "src/externals/service_dispatch.h"
#include "src/data_management/data_conversion_cpu.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace data_management
{
namespace internal
{
/* only for AVX512 architecture with using intrinsics */
#if defined(DAAL_INTEL_CPP_COMPILER)
template <typename T>
static bool tryToCopyFuncAVX512(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    typedef void (*funcType)(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets);
    static funcType ptr = NULL;

    if (!ptr)
    {
        daal::services::Environment * env = daal::services::Environment::getInstance();
        if (!env) return false; // Environment not initialized
        int cpuid = (int)env->getCpuId();

        switch (cpuid)
        {
    #ifdef DAAL_KERNEL_AVX512
        case avx512: DAAL_KERNEL_AVX512_ONLY_CODE(ptr = vectorCopy<T, avx512>); break;
    #endif
        default: return false;
        }
    }

    ptr(nrows, ncols, dst, ptrMin, arrOffsets);
    return true;
}
#else
template <typename T>
static bool tryToCopyFuncAVX512(const size_t nrows, const size_t ncols, void * dst, void const * ptrMin, DAAL_INT64 const * arrOffsets)
{
    return false;
}
#endif

template <typename T1, typename T2>
static void vectorConvertFunc(size_t n, const void * src, void * dst)
{
#define DAAL_VECTOR_CONVERT_CPU(cpuId, ...) vectorConvertFuncCpu<T1, T2, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_VECTOR_CONVERT_CPU, n, src, dst);

#undef DAAL_VECTOR_CONVERT_CPU
}

template <typename T1, typename T2>
static void vectorStrideConvertFunc(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride)
{
#define DAAL_VECTOR_STRIDE_CONVERT_CPU(cpuId, ...) vectorStrideConvertFuncCpu<T1, T2, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_VECTOR_STRIDE_CONVERT_CPU, n, src, srcByteStride, dst, dstByteStride);

#undef DAAL_VECTOR_STRIDE_CONVERT_CPU
}

template <typename T>
DAAL_EXPORT void vectorAssignValueToArray(T * const dataPtr, const size_t n, const T value)
{
#define DAAL_VECTOR_ASSIGN_VALUE_TO_ARRAY_CPU(cpuId, ...) vectorAssignValueToArrayCpu<T, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_VECTOR_ASSIGN_VALUE_TO_ARRAY_CPU, dataPtr, n, &value);

#undef DAAL_VECTOR_ASSIGN_VALUE_TO_ARRAY_CPU
}

#define DAAL_REGISTER_VECTOR_ASSIGN(Type) \
    template DAAL_EXPORT void vectorAssignValueToArray<Type>(Type * const ptr, const size_t n, const Type value);
DAAL_REGISTER_WITH_HOMOGEN_NT_TYPES(DAAL_REGISTER_VECTOR_ASSIGN)

#undef DAAL_TABLE_UP_ENTRY
#define DAAL_TABLE_UP_ENTRY(F, T)            \
    {                                        \
        F<T, float>, F<T, double>, F<T, int> \
    }

#undef DAAL_TABLE_DOWN_ENTRY
#define DAAL_TABLE_DOWN_ENTRY(F, T)          \
    {                                        \
        F<float, T>, F<double, T>, F<int, T> \
    }

#undef DAAL_CONVERT_UP_TABLE
#define DAAL_CONVERT_UP_TABLE(F)                                                                                                          \
    {                                                                                                                                     \
        DAAL_TABLE_UP_ENTRY(F, float), DAAL_TABLE_UP_ENTRY(F, double), DAAL_TABLE_UP_ENTRY(F, int), DAAL_TABLE_UP_ENTRY(F, unsigned int), \
            DAAL_TABLE_UP_ENTRY(F, DAAL_INT64), DAAL_TABLE_UP_ENTRY(F, DAAL_UINT64), DAAL_TABLE_UP_ENTRY(F, char),                        \
            DAAL_TABLE_UP_ENTRY(F, unsigned char), DAAL_TABLE_UP_ENTRY(F, short), DAAL_TABLE_UP_ENTRY(F, unsigned short),                 \
    }

#undef DAAL_CONVERT_DOWN_TABLE
#define DAAL_CONVERT_DOWN_TABLE(F)                                                                                                                \
    {                                                                                                                                             \
        DAAL_TABLE_DOWN_ENTRY(F, float), DAAL_TABLE_DOWN_ENTRY(F, double), DAAL_TABLE_DOWN_ENTRY(F, int), DAAL_TABLE_DOWN_ENTRY(F, unsigned int), \
            DAAL_TABLE_DOWN_ENTRY(F, DAAL_INT64), DAAL_TABLE_DOWN_ENTRY(F, DAAL_UINT64), DAAL_TABLE_DOWN_ENTRY(F, char),                          \
            DAAL_TABLE_DOWN_ENTRY(F, unsigned char), DAAL_TABLE_DOWN_ENTRY(F, short), DAAL_TABLE_DOWN_ENTRY(F, unsigned short),                   \
    }

template <typename T>
DAAL_EXPORT vectorCopy2vFuncType getVector()
{
    return tryToCopyFuncAVX512<T>;
}

template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<float>()
{
    return tryToCopyFuncAVX512<float>;
}

template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<double>()
{
    return tryToCopyFuncAVX512<double>;
}

template <>
DAAL_EXPORT vectorCopy2vFuncType getVector<int>()
{
    return NULL; /* no implementation for integer */
}

DAAL_EXPORT vectorConvertFuncType getVectorUpCast(int idx1, int idx2)
{
    static vectorConvertFuncType table[][3] = DAAL_CONVERT_UP_TABLE(vectorConvertFunc);
    DAAL_ASSERT(idx1 * sizeof table[0] < sizeof table);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorConvertFuncType getVectorDownCast(int idx1, int idx2)
{
    static vectorConvertFuncType table[][3] = DAAL_CONVERT_DOWN_TABLE(vectorConvertFunc);
    DAAL_ASSERT(idx1 * sizeof table[0] < sizeof table);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideUpCast(int idx1, int idx2)
{
    static vectorStrideConvertFuncType table[][3] = DAAL_CONVERT_UP_TABLE(vectorStrideConvertFunc);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideDownCast(int idx1, int idx2)
{
    static vectorStrideConvertFuncType table[][3] = DAAL_CONVERT_DOWN_TABLE(vectorStrideConvertFunc);
    return table[idx1][idx2];
}

} // namespace internal
namespace data_feature_utils
{
DAAL_EXPORT internal::vectorConvertFuncType getVectorUpCast(int idx1, int idx2)
{
    return internal::getVectorUpCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorConvertFuncType getVectorDownCast(int idx1, int idx2)
{
    return internal::getVectorDownCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorStrideConvertFuncType getVectorStrideUpCast(int idx1, int idx2)
{
    return internal::getVectorStrideUpCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorStrideConvertFuncType getVectorStrideDownCast(int idx1, int idx2)
{
    return internal::getVectorStrideDownCast(idx1, idx2);
}

} // namespace data_feature_utils

} // namespace data_management
} // namespace daal
