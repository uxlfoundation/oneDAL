/* file: low_order_moments_container.h */
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
//  Implementation of low order moments calculation algorithm container.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_CONTAINER_H__
#define __LOW_ORDER_MOMENTS_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/moments/low_order_moments_batch.h"
#include "algorithms/moments/low_order_moments_online.h"
#include "algorithms/moments/low_order_moments_distributed.h"
#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * dataTable = input->get(data).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTable, result, par);
}

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
