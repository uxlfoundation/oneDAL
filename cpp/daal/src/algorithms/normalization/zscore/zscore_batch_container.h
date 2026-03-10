/* file: zscore_batch_container.h */
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
//  Implementation of zscore normalization calculation algorithm container.
//--
*/

#include "algorithms/normalization/zscore.h"
#include "src/algorithms/normalization/zscore/zscore_base.h"
#include "src/algorithms/normalization/zscore/zscore_dense_default_kernel.h"
#include "src/algorithms/normalization/zscore/zscore_dense_sum_kernel.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the z-score normalization algorithm.
 *        It is associated with the daal::algorithms::normalization::zscore::Batch class
 *        and supports methods of z-score normalization computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the z-score normalization algorithms, double or float
 * \tparam method           Z-score normalization computation method, daal::algorithms::normalization::zscore::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the z-score normalization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the z-score normalization algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ZScoreKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                          = static_cast<Input *>(_in);
    Result * result                        = static_cast<Result *>(_res);
    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    NumericTablePtr inputTable      = input->get(data);
    NumericTablePtr resultTable     = result->get(normalizedData);
    NumericTablePtr resultMeans     = result->get(means);
    NumericTablePtr resultVariances = result->get(variances);

    if (method == defaultDense)
    {
        Parameter<algorithmFPType, defaultDense> * parameter = static_cast<Parameter<algorithmFPType, defaultDense> *>(par);
        parameter->moments->input.set(low_order_moments::data, inputTable);
    }

    __DAAL_CALL_KERNEL(env, internal::ZScoreKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputTable, *resultTable,
                       *resultMeans, *resultVariances, *par);
}

} // namespace internal

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
