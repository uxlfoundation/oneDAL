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

#include "src/services/internal/daal_kernel_defines.h"

#if defined(TARGET_X86_64)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                                                        \
        switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID,                                                                 \
                                  static_cast<daal::internal::CpuType>(daal::services::Environment::getInstance()->getCpuId())))            \
        {                                                                                                                                   \
            DAAL_KERNEL_SSE42_ONLY_CODE(case daal::internal::sse42 : func(daal::internal::sse42, __VA_ARGS__); break;)    \
            DAAL_KERNEL_AVX2_ONLY_CODE(case daal::internal::avx2 : func(daal::internal::avx2, __VA_ARGS__); break;)       \
            DAAL_KERNEL_AVX512_ONLY_CODE(case daal::internal::avx512 : func(daal::internal::avx512, __VA_ARGS__); break;) \
            DAAL_EXPAND(default : func(daal::internal::sse2, __VA_ARGS__); break;)                                                 \
        }

#elif defined(TARGET_ARM)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                                                \
        switch (static_cast<daal::internal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))                       \
        {                                                                                                                           \
            DAAL_KERNEL_SVE_ONLY_CODE(case daal::internal::sve : func(daal::internal::sve, __VA_ARGS__); break;)  \
        }

#elif defined(TARGET_RISCV64)

    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                                                    \
        switch (static_cast<daal::internal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))                           \
        {                                                                                                                               \
            DAAL_KERNEL_RV64_ONLY_CODE(case daal::internal::rv64 : func(daal::internal::rv64, __VA_ARGS__); break;)   \
        }

#endif

#endif // __SERVICE_DISPATCH_H__
