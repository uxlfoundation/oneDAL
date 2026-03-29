/* file: kernel_function_polynomial_batch_container.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace polynomial
{
namespace internal
{
using namespace daal::data_management;

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KernelImplPolynomial, method, algorithmFPType);
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

    NumericTable * a[2];
    a[0] = static_cast<NumericTable *>(input->get(X).get());
    a[1] = static_cast<NumericTable *>(input->get(Y).get());

    NumericTable * r[1];
    r[0] = static_cast<NumericTable *>(result->get(values).get());

    const ParameterBase * par        = static_cast<const ParameterBase *>(_par);
    services::Environment::env & env = *_env;

    KernelParameter kernelPar;
    kernelPar.rowIndexX       = par->rowIndexX;
    kernelPar.rowIndexY       = par->rowIndexY;
    kernelPar.rowIndexResult  = par->rowIndexResult;
    kernelPar.computationMode = par->computationMode;
    kernelPar.scale           = static_cast<const Parameter *>(par)->scale;
    kernelPar.shift           = static_cast<const Parameter *>(par)->shift;
    kernelPar.degree          = static_cast<const Parameter *>(par)->degree;
    kernelPar.kernelType      = static_cast<const Parameter *>(par)->kernelType;

    __DAAL_CALL_KERNEL(env, internal::KernelImplPolynomial, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a[0], a[1], r[0], &kernelPar);
}

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
