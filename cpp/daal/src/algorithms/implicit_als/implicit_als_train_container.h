/* file: implicit_als_train_container.h */
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
//  Implementation of implicit ALS training algorithm container.
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_CONTAINER_H__
#define __IMPLICIT_ALS_TRAIN_CONTAINER_H__

#include "algorithms/implicit_als/implicit_als_training_batch.h"
#include "algorithms/implicit_als/implicit_als_training_distributed.h"
#include "src/algorithms/implicit_als/implicit_als_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
/**
 *  \brief Initialize list of implicit ALS training algorithm
 *  kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of implicit ALS model-based training
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for implicit ALS model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of implicit ALS model-based training in the batch processing mode
     */
    services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods to compute the result of implicit ALS model-based training
 * in the distributed processing mode
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : TrainingContainerIface<batch>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSTrainBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * a0 = static_cast<NumericTable *>(input->get(data).get());
    Model * a1        = static_cast<Model *>(input->get(inputModel).get());
    Model * r         = static_cast<Model *>(result->get(model).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, a0, a1, r, par);
}

/**
 *  \brief Initialize list of implicit ALS training algorithm for distributed computing mode
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSTrainDistrStep1Kernel, algorithmFPType);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> * input          = static_cast<DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 * partialResult = static_cast<DistributedPartialResultStep1 *>(_pres);

    PartialModel * pModel       = static_cast<PartialModel *>(input->get(partialModel).get());
    NumericTable * crossProduct = static_cast<NumericTable *>(partialResult->get(outputOfStep1ForStep2).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSTrainDistrStep1Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, pModel, crossProduct, par);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

/**
 *  \brief Initialize list of implicit ALS training algorithm for distributed computing mode
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSTrainDistrStep2Kernel, algorithmFPType);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> * input         = static_cast<DistributedInput<step2Master> *>(_in);
    DistributedPartialResultStep2 * partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    DataCollection * crossProductCollection = static_cast<DataCollection *>(input->get(inputOfStep2FromStep1).get());
    size_t nParts                           = crossProductCollection->size();
    NumericTable ** partialCrossProducts =
        (NumericTable **)daal::services::internal::service_calloc<NumericTable *, cpu>(nParts * sizeof(NumericTable *));
    if (!partialCrossProducts) return services::Status(services::ErrorMemoryAllocationFailed);

    for (size_t i = 0; i < nParts; i++)
    {
        partialCrossProducts[i] = static_cast<NumericTable *>((*crossProductCollection)[i].get());
    }
    NumericTable * crossProduct = static_cast<NumericTable *>(partialResult->get(outputOfStep2ForStep4).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::ImplicitALSTrainDistrStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute,
                                                   nParts, partialCrossProducts, crossProduct, par);

    crossProductCollection->clear();
    daal::services::daal_free(partialCrossProducts);
    partialCrossProducts = nullptr;
    return s;
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

/**
 *  \brief Initialize list of implicit ALS training algorithm for distributed computing mode
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSTrainDistrStep3Kernel, algorithmFPType);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step3Local> * input          = static_cast<DistributedInput<step3Local> *>(_in);
    DistributedPartialResultStep3 * partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    PartialModel * pModel           = static_cast<PartialModel *>(input->get(partialModel).get());
    NumericTable * offsetTable      = static_cast<NumericTable *>(input->get(offset).get());
    KeyValueDataCollection * models = static_cast<KeyValueDataCollection *>(partialResult->get(outputOfStep3ForStep4).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSTrainDistrStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, pModel, offsetTable,
                       models, par);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

/**
 *  \brief Initialize list of implicit ALS training algorithm for distributed computing mode
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSTrainDistrStep4Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step4Local> * input          = static_cast<DistributedInput<step4Local> *>(_in);
    DistributedPartialResultStep4 * partialResult = static_cast<DistributedPartialResultStep4 *>(_pres);

    KeyValueDataCollection * models = static_cast<KeyValueDataCollection *>(input->get(partialModels).get());
    NumericTable * dataTable        = static_cast<NumericTable *>(input->get(partialData).get());
    NumericTable * cpTable          = static_cast<NumericTable *>(input->get(inputOfStep4FromStep2).get());

    PartialModel * partialModel = static_cast<PartialModel *>(partialResult->get(outputOfStep4ForStep1).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::ImplicitALSTrainDistrStep4Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                                                   compute, models, dataTable, cpTable, partialModel, par);

    models->clear();
    return s;
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // namespace internal

} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
