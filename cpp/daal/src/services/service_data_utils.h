/* file: service_data_utils.h */
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
//  Declaration of service constants
//--
*/

#ifndef __SERVICE_DATA_UTILS_H__
#define __SERVICE_DATA_UTILS_H__

#include "src/services/cpu_type.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace services
{
namespace internal
{
using CpuType = daal::internal::CpuType;
template <typename T>
struct MaxVal
{
    DAAL_FORCEINLINE static constexpr T get() { return 0; }
};

template <>
struct MaxVal<int>
{
    DAAL_FORCEINLINE static constexpr int get() { return INT_MAX; }
};

template <>
struct MaxVal<DAAL_INT>
{
    DAAL_FORCEINLINE static constexpr DAAL_INT get()
    {
#if defined(_WIN64) || defined(__x86_64__)
        return INT64_MAX;
#else
        return INT32_MAX;
#endif
    }
};

template <>
struct MaxVal<double>
{
    DAAL_FORCEINLINE static constexpr double get() { return DBL_MAX; }
};

template <>
struct MaxVal<uint32_t>
{
    DAAL_FORCEINLINE static constexpr uint32_t get() { return UINT32_MAX; }
};

template <>
struct MaxVal<float>
{
    DAAL_FORCEINLINE static constexpr float get() { return FLT_MAX; }
};

template <typename T>
struct MinVal
{
    DAAL_FORCEINLINE static constexpr T get() { return 0; }
};

template <>
struct MinVal<int>
{
    DAAL_FORCEINLINE static constexpr int get() { return INT_MIN; }
};

template <>
struct MinVal<double>
{
    DAAL_FORCEINLINE static constexpr double get() { return DBL_MIN; }
};

template <>
struct MinVal<float>
{
    DAAL_FORCEINLINE static constexpr float get() { return FLT_MIN; }
};

template <typename T>
struct EpsilonVal
{
    DAAL_FORCEINLINE static T get() { return 0; }
};

template <>
struct EpsilonVal<double>
{
    DAAL_FORCEINLINE static double get() { return DBL_EPSILON; }
};

template <>
struct EpsilonVal<float>
{
    DAAL_FORCEINLINE static float get() { return FLT_EPSILON; }
};

template <typename T, CpuType cpu>
struct SignBit;

template <CpuType cpu>
struct SignBit<float, cpu>
{
    static int get(float val) { return ((_daal_sp_union_t *)&val)->bits.sign; }
};

template <CpuType cpu>
struct SignBit<double, cpu>
{
    static int get(double val) { return ((_daal_dp_union_t *)&val)->bits.sign; }
};

template <typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, void * src, void * dst);

template <typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, void * src, size_t srcByteStride, void * dst, size_t dstByteStride);

} // namespace internal
} // namespace services
} // namespace daal

#endif
