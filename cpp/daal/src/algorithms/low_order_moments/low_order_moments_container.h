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
namespace internal
{
using namespace daal::internal;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the low order moments algorithm.
 *        This class is associated with daal::algorithms::low_order_moments::Batch class
 *
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::low_order_moments::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the low order moments algorithm in the batch processing mode
     */
    services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__ONLINECONTAINER"></a>
 * \brief Provides methods to run implementations of the low order moments algorithm.
 *        This class is associated with daal::algorithms::low_order_moments::Online class

 *
 * \tparam method           Computation method for the low order moments algorithm, \ref daal::algorithms::low_order_moments::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public daal::algorithms::AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();
    /**
     * Computes a partial result of the low order moments algorithm
     * in the online processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the low order moments algorithm
     * in the online processing mode
     */
    services::Status finalizeCompute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDCONTAINER_STEP_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods to run implementations of the low order moments algorithm in the distributed processing mode.
 *        This class is associated with daal::algorithms::low_order_moments::Distributed class
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};
/**
 * \brief Provides methods to run implementations of the second step of the low order moments algorithm
 *        in the distributed processing mode.
 *        This class is associated with daal::algorithms::low_order_moments::Distributed class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the low order moments, double or float
 * \tparam method           Computation method, \ref daal::algorithms::low_order_moments::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the low order moments algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the low order moments algorithm
     * in the second step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the low order moments algorithm
     * in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

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

template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsOnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    bool isOnline                 = true;
    Input * input                 = static_cast<Input *>(_in);
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);

    NumericTable * dataTable = input->get(data).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTable,
                       partialResult, par, isOnline);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    result->set(minimum, partialResult->get(partialMinimum));
    result->set(maximum, partialResult->get(partialMaximum));
    result->set(sum, partialResult->get(partialSum));
    result->set(sumSquares, partialResult->get(partialSumSquares));
    result->set(sumSquaresCentered, partialResult->get(partialSumSquaresCentered));

    NumericTable * nObservationsTable = partialResult->get(nObservations).get();
    NumericTable * sumTable           = partialResult->get(partialSum).get();
    NumericTable * sumSqTable         = partialResult->get(partialSumSquares).get();
    NumericTable * sumSqCenTable      = partialResult->get(partialSumSquaresCentered).get();

    NumericTable * meanTable      = result->get(mean).get();
    NumericTable * raw2MomTable   = result->get(secondOrderRawMoment).get();
    NumericTable * varianceTable  = result->get(variance).get();
    NumericTable * stDevTable     = result->get(standardDeviation).get();
    NumericTable * variationTable = result->get(variation).get();

    __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute,
                       nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable, varianceTable, stDevTable, variationTable,
                       par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsDistributedKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    PartialResult * partialResult                = static_cast<PartialResult *>(_pres);
    DistributedInput<step2Master> * input        = static_cast<DistributedInput<step2Master> *>(_in);
    data_management::DataCollection * collection = input->get(low_order_moments::partialResults).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    services::Status s;

    s = __DAAL_CALL_KERNEL_STATUS(env, internal::LowOrderMomentsDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                  collection, partialResult, par);

    collection->clear();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    result->set(minimum, partialResult->get(partialMinimum));
    result->set(maximum, partialResult->get(partialMaximum));
    result->set(sum, partialResult->get(partialSum));
    result->set(sumSquares, partialResult->get(partialSumSquares));
    result->set(sumSquaresCentered, partialResult->get(partialSumSquaresCentered));

    services::Status s;

    NumericTable * nObservationsTable = partialResult->get(nObservations).get();
    NumericTable * sumTable           = partialResult->get(partialSum).get();
    NumericTable * sumSqTable         = partialResult->get(partialSumSquares).get();
    NumericTable * sumSqCenTable      = partialResult->get(partialSumSquaresCentered).get();

    NumericTable * meanTable      = result->get(mean).get();
    NumericTable * raw2MomTable   = result->get(secondOrderRawMoment).get();
    NumericTable * varianceTable  = result->get(variance).get();
    NumericTable * stDevTable     = result->get(standardDeviation).get();
    NumericTable * variationTable = result->get(variation).get();

    s = __DAAL_CALL_KERNEL_STATUS(env, internal::LowOrderMomentsDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute,
                                  nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable, varianceTable, stDevTable,
                                  variationTable, par);

    return s;
}

} // namespace internal

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
