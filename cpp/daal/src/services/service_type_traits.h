/* file: service_type_traits.h */
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

#ifndef __SERVICE_TYPE_TRAITS_H__
#define __SERVICE_TYPE_TRAITS_H__

#include "src/services/cpu_type.h"

namespace daal
{
namespace services
{
namespace internal
{
using CpuType = daal::internal::CpuType;
template <CpuType cpu, typename T>
struct RemoveReference
{
    typedef T type;
};

template <CpuType cpu, typename T>
struct RemoveReference<cpu, T &>
{
    typedef T type;
};

template <CpuType cpu, typename T>
struct RemoveReference<cpu, T &&>
{
    typedef T type;
};

template <CpuType cpu, bool templateValue>
struct BoolConstant
{
#if (_MSC_VER <= 1800)
    static const bool value = templateValue;
#else
    static constexpr bool value = templateValue;
#endif
};

template <CpuType cpu>
using TrueConstant = BoolConstant<cpu, true>;

template <CpuType cpu>
using FalseConstant = BoolConstant<cpu, false>;

// Mark all types as non-primitive
template <typename T, CpuType cpu>
struct IsPrimitiveType : FalseConstant<cpu>
{};

// Mark all pointer types as primitive
template <typename T, CpuType cpu>
struct IsPrimitiveType<T *, cpu> : TrueConstant<cpu>
{};

#define __DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(type)       \
    template <CpuType cpu>                                \
    struct IsPrimitiveType<type, cpu> : TrueConstant<cpu> \
    {};

// Mark built-in types as primitive
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(bool);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(char);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(signed char);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(unsigned char);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(short);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(unsigned short);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(int);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(unsigned int);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(long);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(unsigned long);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(long long);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(unsigned long long);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(float);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(double);
__DAAL_INTERNAL_DEFINE_PRIMITIVE_TYPE(long double);

} // namespace internal
} // namespace services
} // namespace daal

#endif
