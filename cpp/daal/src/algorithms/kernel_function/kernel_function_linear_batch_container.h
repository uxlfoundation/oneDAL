/* file: kernel_function_linear_batch_container.h */
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
//  Implementation of kernel function container.
//--
*/

#include "src/algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial_dense_default_kernel.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_polynomial_csr_fast_kernel.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
namespace poly = daal::algorithms::kernel_function::polynomial::internal;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the linear kernel function algorithm.
 *        This class is associated with the Batch class
 *        and supports the method for computing linear kernel functions in the %batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of kernel functions, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 */

template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the linear kernel function algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the linear kernel function algorithm in the batch processing mode
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(poly::KernelImplPolynomial, (method == defaultDense) ? poly::defaultDense : poly::fastCSR, algorithmFPType);
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
    kernelPar.scale           = static_cast<const Parameter *>(par)->k;
    kernelPar.shift           = static_cast<const Parameter *>(par)->b;
    kernelPar.degree          = 1;
    kernelPar.kernelType      = KernelType::linear;

    __DAAL_CALL_KERNEL(env, poly::KernelImplPolynomial,
                       __DAAL_KERNEL_ARGUMENTS((method == defaultDense) ? poly::defaultDense : poly::fastCSR, algorithmFPType), compute, a[0], a[1],
                       r[0], &kernelPar);
}

} // namespace internal

} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
