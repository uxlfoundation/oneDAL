/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#pragma once

#include <daal/include/services/daal_defines.h>

#ifdef __ONEDAL_IDE_MODE__
// If this file is openned in IDE it will complain about
// `_onedal_dispatcher_cpu.hpp` as this file is generated at build time.
// It's recommended to define __ONEDAL_IDE_MODE__ in your IDE settings to
// enable this branch for preprocessor.

#if defined(TARGET_X86_64)
#define ONEDAL_CPU_DISPATCH_SSE42
#define ONEDAL_CPU_DISPATCH_AVX2
#define ONEDAL_CPU_DISPATCH_AVX512
#elif defined(TARGET_ARM)
#define ONEDAL_CPU_DISPATCH_A8SVE
#elif defined(TARGET_RISCV64)
#define ONEDAL_CPU_DISPATCH_RV64
#endif
#else
// This file is automatically generated by build system
#include "oneapi/dal/_dal_cpu_dispatcher_gen.hpp"
#endif

#ifdef ONEDAL_CPU_DISPATCH_SSE42
#define ONEDAL_IF_CPU_DISPATCH_SSE42(x) x
#else
#define ONEDAL_IF_CPU_DISPATCH_SSE42(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX2
#define ONEDAL_IF_CPU_DISPATCH_AVX2(x) x
#else
#define ONEDAL_IF_CPU_DISPATCH_AVX2(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX512
#define ONEDAL_IF_CPU_DISPATCH_AVX512(x) x
#else
#define ONEDAL_IF_CPU_DISPATCH_AVX512(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_A8SVE
#define ONEDAL_IF_CPU_DISPATCH_A8SVE(x) x
#else
#define ONEDAL_IF_CPU_DISPATCH_A8SVE(x)
#endif

#ifdef ONEDAL_CPU_DISPATCH_RV64
#define ONEDAL_IF_CPU_DISPATCH_RV64(x) x
#else
#define ONEDAL_IF_CPU_DISPATCH_RV64(x)
#endif

