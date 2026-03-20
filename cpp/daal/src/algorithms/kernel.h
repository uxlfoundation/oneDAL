/* file: kernel.h */
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
//  Defines used for kernel allocation, deallocation and calling kernel methods
//--
*/

#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "algorithms/algorithm_kernel.h"
#include "services/daal_defines.h"
#include "src/services/service_defines.h"
#include "src/services/internal/daal_internal_kernel_defines.h"

#include "src/algorithms/kernel_config.h"

#undef __DAAL_INITIALIZE_KERNELS
#define __DAAL_INITIALIZE_KERNELS(KernelClass, ...)    \
    {                                                  \
        _kernel = (new KernelClass<__VA_ARGS__, cpu>); \
    }

#undef __DAAL_DEINITIALIZE_KERNELS
#define __DAAL_DEINITIALIZE_KERNELS() \
    {                                 \
        if (_kernel) delete _kernel;  \
    }

#undef __DAAL_KERNEL_ARGUMENTS
#define __DAAL_KERNEL_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_CALL_KERNEL
#define __DAAL_CALL_KERNEL(env, KernelClass, templateArguments, method, ...)            \
    {                                                                                   \
        return ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__); \
    }

#undef __DAAL_CALL_KERNEL_STATUS
#define __DAAL_CALL_KERNEL_STATUS(env, KernelClass, templateArguments, method, ...) \
    ((KernelClass<templateArguments, cpu> *)(_kernel))->method(__VA_ARGS__);

#endif
