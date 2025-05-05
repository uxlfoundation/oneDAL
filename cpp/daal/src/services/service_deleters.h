/* file: service_deleters.h */
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

#ifndef __SERVICE_DELETERS_H__
#define __SERVICE_DELETERS_H__

#include <type_traits>
#include "services/cpu_type.h"

namespace daal
{
namespace services
{
namespace internal
{

/* CPU specific deleters */

template <typename T, CpuType cpu>
struct DefaultDeleter
{
    DefaultDeleter() = default;

    template <typename U>
    DefaultDeleter(const DefaultDeleter<U, cpu> & other)
    {
        static_assert(std::is_base_of<T, U>::value, "U must derive from T to use DefaultDeleter<T, cpu>(const DefaultDeleter<U, cpu> &)");
    }

    template <typename U>
    DefaultDeleter & operator=(const DefaultDeleter<U, cpu> & other)
    {
        static_assert(std::is_base_of<T, U>::value, "U must derive from T to use DefaultDeleter<T, cpu>::operator=(const DefaultDeleter<U, cpu> &)");
    }

    void operator()(T * ptr) { delete ptr; }

    ~DefaultDeleter() = default;
};

template <typename T, CpuType cpu>
struct EmptyDeleter
{
    EmptyDeleter() = default;

    template <typename U>
    EmptyDeleter(const EmptyDeleter<U, cpu> & other)
    {
        static_assert(std::is_base_of<T, U>::value, "U must derive from T to use EmptyDeleter<T, cpu>(const EmptyDeleter<U, cpu> &)");
    }

    template <typename U>
    EmptyDeleter & operator=(const EmptyDeleter<U, cpu> & other)
    {
        static_assert(std::is_base_of<T, U>::value, "U must derive from T to use EmptyDeleter<T, cpu>::operator=(const EmptyDeleter<U, cpu> &)");
    }

    void operator()(T * ptr) {}

    ~EmptyDeleter() = default;
};

} // namespace internal
} // namespace services
} // namespace daal
#endif // __SERVICE_DELETERS_H__
