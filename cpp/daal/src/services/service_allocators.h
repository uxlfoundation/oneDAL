/* file: service_allocators.h */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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

#ifndef __SERVICE_ALLOCATORS_H__
#define __SERVICE_ALLOCATORS_H__

#include "src/services/service_utils.h"
#include "src/externals/service_memory.h"
#include "src/services/service_type_traits.h"

namespace daal
{
namespace services
{
namespace internal
{
template <typename T, CpuType cpu, typename... Args>
inline void constructAt(T * ptr, Args &&... args)
{
    ::new (static_cast<void *>(ptr)) T(forward<cpu, Args>(args)...);
}

template <typename T, CpuType cpu>
inline void destroyAt(T * ptr)
{
    ptr->~T();
}

template <typename T, CpuType cpu, typename... Args>
void constructRange(T * first, T * last, Args &&... args)
{
    for (T * ptr = first; ptr != last; ++ptr)
    {
        constructAt<T, cpu, Args...>(ptr, args...);
    }
}

template <typename T, CpuType cpu>
void destroyRange(T * first, T * last)
{
    for (T * ptr = first; ptr != last; ++ptr)
    {
        destroyAt<T, cpu>(ptr);
    }
}

/* CPU specific allocators */

template <typename T, CpuType cpu>
struct DefaultAllocator
{
    static T * allocate(size_t n) { return new T[n]; }
    static void deallocate(T * ptr) { delete[] ptr; }
};

template <typename T, CpuType cpu>
struct DAALMalloc
{
    static T * allocate(size_t n) { return service_malloc<T, cpu>(n); }
    static void deallocate(T * ptr) { service_free<T, cpu>(ptr); }
};

template <typename T, CpuType cpu>
struct DAALCalloc
{
    static T * allocate(size_t n) { return service_calloc<T, cpu>(n); }
    static void deallocate(T * ptr) { service_free<T, cpu>(ptr); }
};

template <typename T, CpuType cpu>
struct ScalableMalloc
{
    static T * allocate(size_t n) { return service_scalable_malloc<T, cpu>(n); }
    static void deallocate(T * ptr) { service_scalable_free<T, cpu>(ptr); }
};

template <typename T, CpuType cpu>
struct ScalableCalloc
{
    static T * allocate(size_t n) { return service_scalable_calloc<T, cpu>(n); }
    static void deallocate(T * ptr) { service_scalable_free<T, cpu>(ptr); }
};

/* Construction policy that does call explicitly constructor/destructor for type T */
template <typename T, CpuType cpu>
struct DoConstruct
{
    static void construct(T * begin, T * end) { constructRange<T, cpu>(begin, end); }

    static void destroy(T * begin, T * end) { destroyRange<T, cpu>(begin, end); }
};

/* Construction policy that doesn't call constructor/destructor */
template <typename T, CpuType cpu>
struct DoNotConstruct
{
    static void construct(T * begin, T * end) {}
    static void destroy(T * begin, T * end) {}
};

/* Allows to detect primitive and non-primitive types automatically
 * and apply an appropriate construction policy  */
template <typename T, CpuType cpu, bool isPrimitive = IsPrimitiveType<T, cpu>::value>
struct DefaultConstructionPolicy
{};

/* Enables DoNotConstruct policy for all primitive types */
template <typename T, CpuType cpu>
struct DefaultConstructionPolicy<T, cpu, /* isPrimitive = */ true> : DoNotConstruct<T, cpu>
{};

/* Enables DoConstruct policy for all 'complex' types */
template <typename T, CpuType cpu>
struct DefaultConstructionPolicy<T, cpu, /* isPrimitive = */ false> : DoConstruct<T, cpu>
{};

} // namespace internal
} // namespace services
} // namespace daal

#endif
