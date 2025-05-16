/* file: service_memory.h */
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
//  Declaration of memory service functions
//--
*/

#ifndef __SERVICE_MEMORY_H__
#define __SERVICE_MEMORY_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/services/service_profiler.h"

#include <climits>     // UINT_MAX
#include <type_traits> // is_trivially_default_constructible

namespace daal
{
namespace services
{
namespace internal
{

/// Initializes block of memory of length `num` with value.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param ptr      Pointer to the block of memory.
/// @param value    Value to initialize the block with.
/// @param num      Number of elements in the block.
template <typename T, CpuType cpu>
void service_memset_seq(T * const ptr, const T value, const size_t num)
{
    /// TODO: 1. Add aligned branch for the case when num is greater than UINT_MAX
    ///          and ptr is aligned.
    ///       2. When possible, split the loop into two parts:
    ///          - the first part uses unaligned stores until the first aligned address,
    ///          - the second part uses aligned stores until the end of the block.
    if (num < UINT_MAX && !((DAAL_UINT64)ptr & DAAL_DEFAULT_ALIGNMENT_MASK))
    {
        /// Use aligned stores
        const unsigned int num32 = static_cast<unsigned int>(num);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (unsigned int i = 0; i < num32; i++)
        {
            ptr[i] = value;
        }
    }
    else
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < num; i++)
        {
            ptr[i] = value;
        }
    }
}

/// Structure that groups memory initialization methods for various data types together
///
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
template <CpuType cpu>
struct ServiceInitializer
{
    /// Initialize block of memory of length `num` with the value returned by the default constructor.
    ///
    /// @tparam T   Data type of the elements in the block.
    ///
    /// @param ptr      Pointer to the block of memory.
    /// @param num      Number of elements in the block.
    template <typename T>
    static void memset_default(T * const ptr, const size_t num)
    {
        service_memset_seq<T, cpu>(ptr, T(), num);
    }

    /// Initialize block of memory of length `num` with zeros.
    ///
    /// @tparam T   Data type of the elements in the block.
    ///
    /// @param ptr      Pointer to the block of memory.
    /// @param num      Number of elements in the block.
    template <typename T>
    static void memset(T * const ptr, const size_t num)
    {
        char * cptr       = (char *)ptr;
        const size_t size = num * sizeof(T);

        service_memset_seq<char, cpu>(cptr, '\0', size);
    }
};

/// Allocates an aligned memory block for the `size` objects of type `T` and initialize it with the value
/// returned by the default constructor if possible, or with zeros.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param size         Number of elements in the block.
/// @param alignment    The address of the allocates memory should be a multiple of this `alignment`.
///
/// @return In case of success, the aligned address of the initialized memory block; NULL pointer otherwise.
template <typename T, CpuType cpu>
T * service_calloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    T * ptr = (T *)daal::services::daal_malloc(size * sizeof(T), alignment);
    if (ptr == NULL)
    {
        return NULL;
    }

    if constexpr (std::is_trivially_default_constructible_v<T>)
    {
        ServiceInitializer<cpu>::template memset_default<T>(ptr, size);
    }
    else
    {
        ServiceInitializer<cpu>::template memset<T>(ptr, size);
    }

    return ptr;
}

/// Allocates an aligned memory block for the `size` objects of type `T` and initialize it with the value
/// returned by the default constructor if possible, or with zeros.
///
/// This saclable variant allocates memory in a way that scales with the number of processors.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param size         Number of elements in the block.
/// @param alignment    The address of the allocates memory should be a multiple of this `alignment`.
///
/// @return In case of success, the aligned address of the initialized memory block; NULL pointer otherwise.
template <typename T, CpuType cpu>
T * service_scalable_calloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    T * ptr = (T *)threaded_scalable_malloc(size * sizeof(T), alignment);
    if (!ptr)
    {
        return nullptr;
    }

    if constexpr (std::is_trivially_default_constructible_v<T>)
    {
        ServiceInitializer<cpu>::template memset_default<T>(ptr, size);
    }
    else
    {
        ServiceInitializer<cpu>::template memset<T>(ptr, size);
    }

    return ptr;
}

/// Allocates an aligned memory block for the `size` objects of type `T`.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param size         Number of elements in the block.
/// @param alignment    The address of the allocates memory should be a multiple of this `alignment`.
///
/// @return In case of success, the aligned address of the uninitialized memory block; NULL pointer otherwise.
template <typename T, CpuType cpu>
T * service_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    return (T *)daal::services::daal_malloc(size * sizeof(T), alignment);
}

/// Frees the memory allocated with `service_malloc` or `service_calloc`.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param ptr  Pointer to the memory block.
template <typename T, CpuType cpu>
void service_free(T * ptr)
{
    daal::services::daal_free(ptr);
}

/// Allocates an aligned memory block for the `size` objects of type `T`.
/// This saclable variant allocates memory in a way that scales with the number of processors.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param size         Number of elements in the block.
/// @param alignment    The address of the allocates memory should be a multiple of this `alignment`.
///
/// @return In case of success, the aligned address of the uninitialized memory block; NULL pointer otherwise.
template <typename T, CpuType cpu>
T * service_scalable_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    return (T *)threaded_scalable_malloc(size * sizeof(T), alignment);
}

/// Frees the memory allocated with `service_scalable_malloc` or `service_scalable_calloc`.
///
/// @tparam T   Data type of the elements in the block.
/// @tparam cpu Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
///
/// @param ptr  Pointer to the memory block.
template <typename T, CpuType cpu>
void service_scalable_free(T * ptr)
{
    threaded_scalable_free(ptr);
}

template <typename T, CpuType cpu>
T * service_memset(T * const ptr, const T value, const size_t num)
{
    DAAL_PROFILER_SERVICE_TASK(daal_service::service_memset);
    const size_t blockSize = 512;
    size_t nBlocks         = num / blockSize;
    if (nBlocks * blockSize < num)
    {
        nBlocks++;
    }

    threader_for(nBlocks, nBlocks, [&](size_t block) {
        size_t end = (block + 1) * blockSize;
        if (end > num)
        {
            end = num;
        }

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = block * blockSize; i < end; i++)
        {
            ptr[i] = value;
        }
    });
    return ptr;
}

/* Initialize block of memory of length num with entries [startValue, ..., startValue + num -1]*/
template <typename T, CpuType cpu>
void service_memset_incrementing(T * const ptr, const T startValue, const size_t num)
{
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < num; i++)
    {
        ptr[i] = startValue + i;
    }
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
