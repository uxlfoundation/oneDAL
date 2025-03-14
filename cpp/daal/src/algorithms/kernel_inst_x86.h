/* file: kernel_inst_x86.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Defines used for kernel allocation, deallocation and calling kernel methods
//--
*/
#include "services/daal_defines.h"
//useage of DAAL_EXPORT is only to not break ABI. It can be removed in the next major release (2026.0)

#ifndef __KERNEL_INST_X86_H__
    #define __KERNEL_INST_X86_H__

    #define __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, ClassName, BaseClassName, GetCpuid, ...)                          \
        DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                     \
        DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                    \
        DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                     \
        DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, __VA_ARGS__)                                                                   \
        namespace interface1                                                                                                            \
        {                                                                                                                               \
        template <>                                                                                                                     \
        DAAL_EXPORT ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__)    \
                                        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX512_CONTAINER(        \
                                            ContainerTemplate, __VA_ARGS__)>::ClassName(daal::services::Environment::env * daalEnv)     \
            : BaseClassName(daalEnv), _cntr(nullptr)                                                                                    \
        {                                                                                                                               \
            GetCpuid switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, cpuid))                                        \
            {                                                                                                                           \
                DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                        \
                DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                         \
                DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, __VA_ARGS__)                                                       \
            default: _cntr = (new ContainerTemplate<__VA_ARGS__, sse2>(daalEnv)); break;                                                \
            }                                                                                                                           \
        }                                                                                                                               \
                                                                                                                                        \
        template class ClassName<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) \
                                           DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)                                   \
                                               DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>;                           \
        }

#endif
