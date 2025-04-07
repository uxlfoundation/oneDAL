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

#include <climits>     // UINT_MAX
#include <type_traits> // is_trivially_default_constructible

namespace daal
{
namespace services
{
namespace internal
{

/// Initialize block of memory of length num with value.
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
    if (num < UINT_MAX && !((DAAL_UINT64)ptr & DAAL_DEFAULT_ALIGNMENT_MASK))
    {
        /// Use aligned stores
        const unsigned int num32 = static_cast<unsigned int>(num);
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (unsigned int i = 0; i < num32; i++)
        {
            ptr[i] = value;
        }
    }
    else
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < num; i++)
        {
            ptr[i] = value;
        }
    }
}

template <CpuType cpu>
struct ServiceInitializer
{
    template <typename T>
    static void memset_default(T * const ptr, const size_t num)
    {
        service_memset_seq<T, cpu>(ptr, T(), num);
    }

    template <typename T>
    static void memset(T * const ptr, const size_t num)
    {
        char * cptr = (char *)ptr;
        size_t size = num * sizeof(T);

        if (size < UINT_MAX && !((DAAL_UINT64)cptr & DAAL_DEFAULT_ALIGNMENT_MASK))
        {
            /// Use aligned stores
            unsigned int size32 = static_cast<unsigned int>(size);
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            PRAGMA_VECTOR_ALIGNED
            for (unsigned int i = 0; i < size32; i++)
            {
                cptr[i] = '\0';
            }
        }
        else
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < size; i++)
            {
                cptr[i] = '\0';
            }
        }
    }
};

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

template <typename T, CpuType cpu>
T * service_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    return (T *)daal::services::daal_malloc(size * sizeof(T), alignment);
}

template <typename T, CpuType cpu>
void service_free(T * ptr)
{
    daal::services::daal_free(ptr);
}

template <typename T, CpuType cpu>
T * service_scalable_malloc(size_t size, size_t alignment = DAAL_MALLOC_DEFAULT_ALIGNMENT)
{
    return (T *)threaded_scalable_malloc(size * sizeof(T), alignment);
}

template <typename T, CpuType cpu>
void service_scalable_free(T * ptr)
{
    threaded_scalable_free(ptr);
}

template <typename T, CpuType cpu>
T * service_memset(T * const ptr, const T value, const size_t num)
{
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

        PRAGMA_IVDEP
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
    PRAGMA_IVDEP
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
