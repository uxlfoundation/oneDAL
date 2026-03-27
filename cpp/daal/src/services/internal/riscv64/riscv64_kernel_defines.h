/* file: riscv64_kernel_defines.h */
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

#ifndef __RISCV64_KERNEL_DEFINES_H__
#define __RISCV64_KERNEL_DEFINES_H__

#define DAAL_KERNEL_RV64

#if defined(DAAL_KERNEL_RV64)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID           daal::internal::rv64
    #define DAAL_KERNEL_RV64_ONLY(something)                   , something
    #define DAAL_KERNEL_RV64_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, daal::internal::rv64, __VA_ARGS__)
    #define DAAL_KERNEL_RV64_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, daal::internal::rv64, __VA_ARGS__);
    #define DAAL_KERNEL_RV64_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, daal::internal::rv64, __VA_ARGS__)
#else
    #define DAAL_KERNEL_RV64_ONLY(something)
    #define DAAL_KERNEL_RV64_ONLY_CODE(...)
    #define DAAL_KERNEL_RV64_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_RV64_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_RV64_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#endif
