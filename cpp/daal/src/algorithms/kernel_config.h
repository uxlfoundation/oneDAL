/* file: kernel_config.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Wrapper for platform-specific kernels
//--
*/

#ifndef __KERNEL_CONFIG_H__
#define __KERNEL_CONFIG_H__

#include "services/daal_defines.h"
#include "src/services/service_defines.h"
#include "services/internal/daal_kernel_defines.h"

#if defined(TARGET_X86_64)
    #include "src/algorithms/kernel_inst_x86.h"
#elif defined(TARGET_ARM)
    #include "src/algorithms/kernel_inst_arm.h"
#elif defined(TARGET_RISCV64)
    #include "src/algorithms/kernel_inst_riscv64.h"
#endif

#define __DAAL_GET_CPUID int cpuid = daalEnv->cpuid;

#define __DAAL_INSTANTIATE_DISPATCH_CONTAINER(ContainerTemplate, Mode, ...) \
    __DAAL_INSTANTIATE_DISPATCH_IMPL(ContainerTemplate, Mode, AlgorithmDispatchContainer, AlgorithmContainerImpl<Mode>, __DAAL_GET_CPUID, __VA_ARGS__)

#endif
