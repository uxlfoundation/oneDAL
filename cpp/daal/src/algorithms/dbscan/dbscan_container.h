/* file: dbscan_container.h */
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
//  Implementation of DBSCAN container.
//--
*/

#ifndef __DBSCAN_CONTAINER_H__
#define __DBSCAN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/dbscan/dbscan_types.h"
#include "algorithms/dbscan/dbscan_batch.h"
#include "algorithms/dbscan/dbscan_distributed.h"
#include "src/algorithms/dbscan/dbscan_kernel.h"

namespace daal
{
namespace algorithms
{
namespace dbscan
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANBatchKernel, algorithmFPType, method);
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

    const NumericTableConstPtr ntData    = input->get(data);
    const NumericTableConstPtr ntWeights = input->get(weights);

    const NumericTablePtr ntAssignments      = result->get(assignments);
    const NumericTablePtr ntNClusters        = result->get(nClusters);
    const NumericTablePtr ntCoreIndices      = result->get(coreIndices);
    const NumericTablePtr ntCoreObservations = result->get(coreObservations);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    if (par->memorySavingMode == false)
    {
        __DAAL_CALL_KERNEL(env, internal::DBSCANBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), computeNoMemSave, ntData.get(),
                           ntWeights.get(), ntAssignments.get(), ntNClusters.get(), ntCoreIndices.get(), ntCoreObservations.get(), par);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::DBSCANBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), computeMemSave, ntData.get(),
                           ntWeights.get(), ntAssignments.get(), ntNClusters.get(), ntCoreIndices.get(), ntCoreObservations.get(), par);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep1Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> * input          = static_cast<DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 * partialResult = static_cast<DistributedPartialResultStep1 *>(_pres);

    const NumericTablePtr ntData = input->get(step1Data);

    const NumericTablePtr ntPartialOrder = partialResult->get(partialOrder);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep1Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, ntData.get(),
                       ntPartialOrder.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Local> * input          = static_cast<DistributedInput<step2Local> *>(_in);
    DistributedPartialResultStep2 * partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    const DataCollectionPtr dcPartialData = input->get(partialData);

    const NumericTablePtr ntBoundingBox = partialResult->get(boundingBox);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialData.get(),
                       ntBoundingBox.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep3Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step3Local> * input          = static_cast<DistributedInput<step3Local> *>(_in);
    DistributedPartialResultStep3 * partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    const DataCollectionPtr dcPartialData          = input->get(partialData);
    const DataCollectionPtr dcPartialBoundingBoxes = input->get(step3PartialBoundingBoxes);

    const NumericTablePtr ntSplit = partialResult->get(split);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialData.get(),
                       dcPartialBoundingBoxes.get(), ntSplit.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep4Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step4Local> * input          = static_cast<DistributedInput<step4Local> *>(_in);
    DistributedPartialResultStep4 * partialResult = static_cast<DistributedPartialResultStep4 *>(_pres);

    const DataCollectionPtr dcPartialData   = input->get(partialData);
    const DataCollectionPtr dcPartialSplits = input->get(step4PartialSplits);
    const DataCollectionPtr dcPartialOrders = input->get(step4PartialOrders);

    const DataCollectionPtr dcPartitionedData          = partialResult->get(partitionedData);
    const DataCollectionPtr dcPartitionedPartialOrders = partialResult->get(partitionedPartialOrders);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep4Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialData.get(),
                       dcPartialSplits.get(), dcPartialOrders.get(), dcPartitionedData.get(), dcPartitionedPartialOrders.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep5Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step5Local> * input          = static_cast<DistributedInput<step5Local> *>(_in);
    DistributedPartialResultStep5 * partialResult = static_cast<DistributedPartialResultStep5 *>(_pres);

    const DataCollectionPtr dcPartialData          = input->get(partialData);
    const DataCollectionPtr dcPartialBoundingBoxes = input->get(step5PartialBoundingBoxes);

    const DataCollectionPtr dcPartitionedHaloData        = partialResult->get(partitionedHaloData);
    const DataCollectionPtr dcPartitionedHaloDataIndices = partialResult->get(partitionedHaloDataIndices);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep5Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialData.get(),
                       dcPartialBoundingBoxes.get(), dcPartitionedHaloData.get(), dcPartitionedHaloDataIndices.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep6Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step6Local> * input          = static_cast<DistributedInput<step6Local> *>(_in);
    DistributedPartialResultStep6 * partialResult = static_cast<DistributedPartialResultStep6 *>(_pres);

    const DataCollectionPtr dcPartialData     = input->get(partialData);
    const DataCollectionPtr dcHaloData        = input->get(haloData);
    const DataCollectionPtr dcHaloDataIndices = input->get(haloDataIndices);
    const DataCollectionPtr dcHaloBlocks      = input->get(haloBlocks);

    const NumericTablePtr ntClusterStructure = partialResult->get(step6ClusterStructure);
    const NumericTablePtr ntFinishedFlag     = partialResult->get(step6FinishedFlag);
    const NumericTablePtr ntNClusters        = partialResult->get(step6NClusters);
    const DataCollectionPtr dcQueries        = partialResult->get(step6Queries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    if (par->memorySavingMode == false)
    {
        __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep6Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), computeNoMemSave,
                           dcPartialData.get(), dcHaloData.get(), dcHaloDataIndices.get(), dcHaloBlocks.get(), ntClusterStructure.get(),
                           ntFinishedFlag.get(), ntNClusters.get(), dcQueries.get(), par);
    }
    else
    {
        __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep6Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), computeMemSave,
                           dcPartialData.get(), dcHaloData.get(), dcHaloDataIndices.get(), dcHaloBlocks.get(), ntClusterStructure.get(),
                           ntFinishedFlag.get(), ntNClusters.get(), dcQueries.get(), par);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step7Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep7Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step7Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step7Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step7Master> * input         = static_cast<DistributedInput<step7Master> *>(_in);
    DistributedPartialResultStep7 * partialResult = static_cast<DistributedPartialResultStep7 *>(_pres);

    const DataCollectionPtr dcPartialFinishedFlags = input->get(partialFinishedFlags);

    const NumericTablePtr ntFinishedFlag = partialResult->get(finishedFlag);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep7Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialFinishedFlags.get(),
                       ntFinishedFlag.get());
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step7Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step8Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep8Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step8Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step8Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step8Local> * input          = static_cast<DistributedInput<step8Local> *>(_in);
    DistributedPartialResultStep8 * partialResult = static_cast<DistributedPartialResultStep8 *>(_pres);

    const NumericTablePtr ntInputClusterStructure = input->get(step8InputClusterStructure);
    const NumericTablePtr ntInputNClusters        = input->get(step8InputNClusters);
    const DataCollectionPtr dcPartialQueries      = input->get(step8PartialQueries);

    const NumericTablePtr ntClusterStructure = partialResult->get(step8ClusterStructure);
    const NumericTablePtr ntFinishedFlag     = partialResult->get(step8FinishedFlag);
    const NumericTablePtr ntNClusters        = partialResult->get(step8NClusters);
    const DataCollectionPtr dcQueries        = partialResult->get(step8Queries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep8Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       ntInputClusterStructure.get(), ntInputNClusters.get(), dcPartialQueries.get(), ntClusterStructure.get(), ntFinishedFlag.get(),
                       ntNClusters.get(), dcQueries.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step8Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step9Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep9Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step9Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step9Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step9Master> * input         = static_cast<DistributedInput<step9Master> *>(_in);
    DistributedPartialResultStep9 * partialResult = static_cast<DistributedPartialResultStep9 *>(_pres);

    const DataCollectionPtr dcPartialNClusters = input->get(partialNClusters);

    const DataCollectionPtr dcClusterOffsets = partialResult->get(clusterOffsets);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep9Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dcPartialNClusters.get(),
                       dcClusterOffsets.get());
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step9Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    DistributedPartialResultStep9 * partialResult = static_cast<DistributedPartialResultStep9 *>(_pres);
    DistributedResultStep9 * result               = static_cast<DistributedResultStep9 *>(_res);

    const DataCollectionPtr dcClusterOffsets = partialResult->get(clusterOffsets);

    const NumericTablePtr ntNClusters = result->get(step9NClusters);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep9Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute,
                       dcClusterOffsets.get(), ntNClusters.get());
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step10Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep10Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step10Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step10Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step10Local> * input          = static_cast<DistributedInput<step10Local> *>(_in);
    DistributedPartialResultStep10 * partialResult = static_cast<DistributedPartialResultStep10 *>(_pres);

    const NumericTablePtr ntInputClusterStructure = input->get(step10InputClusterStructure);
    const NumericTablePtr ntClusterOffset         = input->get(step10ClusterOffset);

    const NumericTablePtr ntClusterStructure = partialResult->get(step10ClusterStructure);
    const NumericTablePtr ntFinishedFlag     = partialResult->get(step10FinishedFlag);
    const DataCollectionPtr dcQueries        = partialResult->get(step10Queries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep10Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       ntInputClusterStructure.get(), ntClusterOffset.get(), ntClusterStructure.get(), ntFinishedFlag.get(), dcQueries.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step10Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step11Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep11Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step11Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step11Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step11Local> * input          = static_cast<DistributedInput<step11Local> *>(_in);
    DistributedPartialResultStep11 * partialResult = static_cast<DistributedPartialResultStep11 *>(_pres);

    const NumericTablePtr ntInputClusterStructure = input->get(step11InputClusterStructure);
    const DataCollectionPtr dcPartialQueries      = input->get(step11PartialQueries);

    const NumericTablePtr ntClusterStructure = partialResult->get(step11ClusterStructure);
    const NumericTablePtr ntFinishedFlag     = partialResult->get(step11FinishedFlag);
    const DataCollectionPtr dcQueries        = partialResult->get(step11Queries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep11Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       ntInputClusterStructure.get(), dcPartialQueries.get(), ntClusterStructure.get(), ntFinishedFlag.get(), dcQueries.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step11Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step12Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep12Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step12Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step12Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step12Local> * input          = static_cast<DistributedInput<step12Local> *>(_in);
    DistributedPartialResultStep12 * partialResult = static_cast<DistributedPartialResultStep12 *>(_pres);

    const NumericTablePtr ntInputClusterStructure = input->get(step12InputClusterStructure);
    const DataCollectionPtr dcPartialOrders       = input->get(step12PartialOrders);

    const DataCollectionPtr dcAssignmentQueries = partialResult->get(assignmentQueries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep12Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       ntInputClusterStructure.get(), dcPartialOrders.get(), dcAssignmentQueries.get(), par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step12Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step13Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::DBSCANDistrStep13Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step13Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step13Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step13Local> * input          = static_cast<DistributedInput<step13Local> *>(_in);
    DistributedPartialResultStep13 * partialResult = static_cast<DistributedPartialResultStep13 *>(_pres);

    const DataCollectionPtr dcPartialAssignmentQueries = input->get(partialAssignmentQueries);

    const NumericTablePtr ntAssignmentQueries = partialResult->get(step13AssignmentQueries);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep13Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       dcPartialAssignmentQueries.get(), ntAssignmentQueries.get(), par);
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step13Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    DistributedPartialResultStep13 * partialResult = static_cast<DistributedPartialResultStep13 *>(_pres);
    DistributedResultStep13 * result               = static_cast<DistributedResultStep13 *>(_res);

    const NumericTablePtr ntAssignmentQueries = partialResult->get(step13AssignmentQueries);

    const NumericTablePtr ntAssignments = result->get(step13Assignments);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DBSCANDistrStep13Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute,
                       ntAssignmentQueries.get(), ntAssignments.get(), par);
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
