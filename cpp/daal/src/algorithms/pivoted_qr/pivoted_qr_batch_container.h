/* file: pivoted_qr_batch_container.h */
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
//  Implementation of pivoted_qr calculation algorithm container.
//--
*/

#ifndef __PIVOTED_QR_BATCH_CONTAINER_H__
#define __PIVOTED_QR_BATCH_CONTAINER_H__

#include "algorithms/pivoted_qr/pivoted_qr_types.h"
#include "algorithms/pivoted_qr/pivoted_qr_batch.h"
#include "src/algorithms/pivoted_qr/pivoted_qr_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the pivoted QR decomposition algorithm
 *
 * \tparam method           Pivoted QR computation method, \ref daal::algorithms::pivoted_qr::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations for the pivoted QR, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the pivoted QR decomposition algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the pivoted QR decomposition algorithm in the batch processing mode
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PivotedQRKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input         = static_cast<Input *>(_in);
    Result * result       = static_cast<Result *>(_res);
    Parameter * parameter = static_cast<Parameter *>(_par);

    NumericTable * dataTable               = static_cast<NumericTable *>(input->get(data).get());
    NumericTable * matrixQTable            = static_cast<NumericTable *>(result->get(matrixQ).get());
    NumericTable * matrixRTable            = static_cast<NumericTable *>(result->get(matrixR).get());
    NumericTable * permutationMatrixTable  = static_cast<NumericTable *>(result->get(permutationMatrix).get());
    NumericTable * permutedColumnsTable    = parameter->permutedColumns.get();
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PivotedQRKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, *dataTable, *matrixQTable,
                       *matrixRTable, *permutationMatrixTable, permutedColumnsTable);
}

} // namespace internal

} //namespace pivoted_qr

} //namespace algorithms

} //namespace daal

#endif
