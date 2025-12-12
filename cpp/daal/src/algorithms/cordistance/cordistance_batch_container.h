/* file: cordistance_batch_container.h */
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
//  Implementation of correlation distance calculation algorithm container.
//--
*/

#ifndef __CORDISTANCE_BATCH_CONTAINER_H__
#define __CORDISTANCE_BATCH_CONTAINER_H__

#include "algorithms/distance/correlation_distance.h"
#include "src/algorithms/cordistance/cordistance_kernel.h"

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
/**
 *  \brief Initialize list of correlation distance, double precision
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistanceKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result * result = static_cast<Result *>(_res);
    Input * input   = static_cast<Input *>(_in);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable * a0                      = static_cast<NumericTable *>(input->get(data).get());
    NumericTable ** a                      = &a0;
    NumericTable * r0                      = static_cast<NumericTable *>(result->get(correlationDistance).get());
    NumericTable ** r                      = &r0;
    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DistanceKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal

#endif
