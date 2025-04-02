/* file: service_dispatch.h */
/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/*
//++
//  Universal dispatcher of abstract function with cpu parameter.
//--
*/

#ifndef __SERVICE_DISPATCH_H__
#define __SERVICE_DISPATCH_H__

#include "services/internal/daal_kernel_defines.h"

#if defined(TARGET_X86_64)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                                    \
        switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID,                                             \
                                  static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId())))  \
        {                                                                                                               \
            DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42 : func(daal::CpuType::sse42, __VA_ARGS__); break;)    \
            DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2 : func(daal::CpuType::avx2, __VA_ARGS__); break;)       \
            DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512 : func(daal::CpuType::avx512, __VA_ARGS__); break;) \
            DAAL_EXPAND(default : func(daal::CpuType::sse2, __VA_ARGS__); break;)                                       \
        }

#elif defined(TARGET_ARM)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                           \
        switch (static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))            \
        {                                                                                                      \
            DAAL_KERNEL_SVE_ONLY_CODE(case daal::CpuType::sve : func(daal::CpuType::sve, __VA_ARGS__); break;) \
        }

#elif defined(TARGET_RISCV64)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                              \
        switch (static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))               \
        {                                                                                                         \
            DAAL_KERNEL_RV64_ONLY_CODE(case daal::CpuType::rv64 : func(daal::CpuType::rv64, __VA_ARGS__); break;) \
        }

#endif

#endif // __SERVICE_DISPATCH_H__
