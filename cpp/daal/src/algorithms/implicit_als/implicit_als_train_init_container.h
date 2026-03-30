/* file: implicit_als_train_init_container.h */
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
//  Implementation of implicit ALS initialization algorithm container.
//--
*/

#ifndef __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__
#define __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__

#include "algorithms/implicit_als/implicit_als_training_init_batch.h"
#include "algorithms/implicit_als/implicit_als_training_init_distributed.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/implicit_als/implicit_als_train_init_kernel.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the implicit ALS initialization algorithm
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes initial values for implicit ALS model-based training in the batch processing mode
     */
    services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the results of the implicit ALS initialization algorithm
 * in the distributed processing mode
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the first step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();
    /**
     * Computes a partial result of the implicit ALS initialization algorithm
     * in the first step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the implicit ALS initialization algorithm
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() override { return services::Status(); }
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods to train the implicit ALS model in the third step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the implicit ALS initialization algorithm with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();
    /**
     * Computes a partial result of the implicit ALS initialization algorithm
     * in the third step of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the implicit ALS initialization algorithm
     * in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() override { return services::Status(); }
};

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : TrainingContainerIface<batch>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * const input   = static_cast<Input *>(_in);
    Result * const result = static_cast<Result *>(_res);

    NumericTable * const a     = input->get(data).get();
    Model * const m            = result->get(training::init::model).get();
    NumericTable * const items = m->getItemsFactors().get();
    NumericTable * const users = m->getUsersFactors().get();

    Parameter * const par                  = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a, items, users, par,
                       *services::dynamicPointerCast<engines::BatchBase>(par->engine));
}

/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitDistrKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedParameter * par           = static_cast<DistributedParameter *>(_par);
    DistributedInput<step1Local> * input = static_cast<DistributedInput<step1Local> *>(_in);
    PartialResult * partialResult        = static_cast<PartialResult *>(_pres);

    NumericTable * dataTable      = input->get(data).get();
    NumericTable * partitionTable = par->partition.get();
    size_t nRows                  = partitionTable->getNumberOfRows();
    size_t nParts                 = nRows - 1;
    if (nParts == 0)
    {
        BlockDescriptor<int> block;
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        int * nPartsData = block.getBlockPtr();
        nParts           = nPartsData[0];
        partitionTable->releaseBlockOfRows(block);
    }

    implicit_als::PartialModel * pModel = partialResult->get(partialModel).get();
    NumericTable * result               = pModel->getFactors().get();

    daal::internal::TArray<NumericTable *, cpu> dataPartsPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> blocksToLocalPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> userOffsetsPtr(nParts);
    KeyValueDataCollection & dataPartsCollection     = *(partialResult->get(outputOfStep1ForStep2));
    KeyValueDataCollection & blocksToLocalCollection = *(partialResult->get(outputOfInitForComputeStep3));
    KeyValueDataCollection & userOffsetsCollection   = *(partialResult->get(offsets));
    for (size_t i = 0; i < nParts; i++)
    {
        dataPartsPtr[i]     = static_cast<NumericTable *>(dataPartsCollection[i].get());
        blocksToLocalPtr[i] = static_cast<NumericTable *>(blocksToLocalCollection[i].get());
        userOffsetsPtr[i]   = static_cast<NumericTable *>(userOffsetsCollection[i].get());
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitDistrKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTable,
                       partitionTable, dataPartsPtr.get(), blocksToLocalPtr.get(), userOffsetsPtr.get(), result, par,
                       *services::dynamicPointerCast<engines::BatchBase>(par->engine));
}

/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitDistrStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Local> * input          = static_cast<DistributedInput<step2Local> *>(_in);
    DistributedPartialResultStep2 * partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    KeyValueDataCollection & dataPartsCollection     = *(input->get(inputOfStep2FromStep1));
    KeyValueDataCollection & blocksToLocalCollection = *(partialResult->get(outputOfInitForComputeStep3));
    KeyValueDataCollection & itemOffsetsCollection   = *(partialResult->get(offsets));
    size_t nParts                                    = dataPartsCollection.size();
    daal::internal::TArray<NumericTable *, cpu> dataPartsPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> blocksToLocalPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> itemOffsetsPtr(nParts);

    for (size_t i = 0; i < nParts; i++)
    {
        dataPartsPtr[i]     = static_cast<NumericTable *>(dataPartsCollection[i].get());
        blocksToLocalPtr[i] = static_cast<NumericTable *>(blocksToLocalCollection[i].get());
        itemOffsetsPtr[i]   = static_cast<NumericTable *>(itemOffsetsCollection[i].get());
    }

    NumericTable * dataTable               = partialResult->get(transposedData).get();
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitDistrStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, nParts,
                       dataPartsPtr.get(), dataTable, blocksToLocalPtr.get(), itemOffsetsPtr.get());
}

} // namespace internal

} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
