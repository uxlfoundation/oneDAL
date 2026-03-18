/* file: daal_kernel_defines.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Common definitions.
//--
*/

#ifndef __DAAL_KERNEL_DEFINES_H__
#define __DAAL_KERNEL_DEFINES_H__

#include "services/env_detect.h"

#define DAAL_KERNEL_SSE42
#define DAAL_KERNEL_AVX2
#define DAAL_KERNEL_AVX512

#define __DAAL_KERNEL_MIN(a, b) ((a) < (b) ? (a) : (b))

#if defined(TARGET_X86_64)
    #include "src/services/internal/x86_64/x86_64_kernel_defines.h"
#elif defined(TARGET_ARM)
    #include "src/services/internal/aarch64/aarch64_kernel_defines.h"
#elif defined(TARGET_RISCV64)
    #include "src/services/internal/riscv64/riscv64_kernel_defines.h"
#endif

#define DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, ...) ContainerTemplate<__VA_ARGS__, cpuType>
#define DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, cpuType, ...)                              \
case cpuType:                                                                                    \
    _cntr = (new DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, __VA_ARGS__)(daalEnv)); \
    break;

#define DAAL_EXPAND(...) __VA_ARGS__

#endif
