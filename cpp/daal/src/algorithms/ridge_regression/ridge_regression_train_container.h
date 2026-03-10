/* file: ridge_regression_train_container.h */
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
//  Implementation of ridge regression container.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_CONTAINER_H__
#define __RIDGE_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/ridge_regression/ridge_regression_training_batch.h"
#include "algorithms/ridge_regression/ridge_regression_training_online.h"
#include "algorithms/ridge_regression/ridge_regression_training_distributed.h"
#include "src/algorithms/ridge_regression/ridge_regression_train_kernel.h"
#include "algorithms/ridge_regression/ridge_regression_ne_model.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for normal equations ridge regression model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for ridge regression model-based training with a specified environment in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~BatchContainer();

    /**
     * Computes the result of ridge regression model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};
/**
 * \brief Class containing methods for ridge regression model-based training
 * in the online processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public TrainingContainerIface<online>
{
public:
    /**
     * Constructs a container for ridge regression model-based training with a specified environment in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of ridge regression model-based training in the online processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;

    /**
     * Computes the result of ridge regression model-based training in the online processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods for ridge regression model-based training in the distributed processing mode
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for ridge regression model-based training in the second step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for ridge regression model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of ridge regression model-based training in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;

    /**
     * Computes the result of ridge regression model-based training in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() override;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * const input        = static_cast<Input *>(_in);
    Result * const result      = static_cast<Result *>(_res);
    TrainParameter * const par = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr m = ridge_regression::ModelNormEq::cast(result->get(model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *(input->get(data)),
                       *(input->get(dependentVariables)), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag,
                       *(par->ridgeParameters));
}

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    Input * const input                 = static_cast<Input *>(_in);
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    TrainParameter * const par          = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr m = ridge_regression::ModelNormEq::cast(partialResult->get(training::partialModel));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *(input->get(data)),
                       *(input->get(dependentVariables)), *(m->getXTXTable()), *(m->getXTYTable()), par->interceptFlag);
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    Result * const result               = static_cast<Result *>(_res);
    TrainParameter * const par          = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(partialResult->get(training::partialModel));
    ridge_regression::ModelNormEqPtr m  = ridge_regression::ModelNormEq::cast(result->get(training::model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, *(pm->getXTXTable()),
                       *(pm->getXTYTable()), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag, *(par->ridgeParameters));
}

/**
 *  \brief Initialize list of ridge regression kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistributedKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> * const input = static_cast<DistributedInput<step2Master> *>(_in);
    PartialResult * const partialResult         = static_cast<PartialResult *>(_pres);

    DataCollectionPtr collection = input->get(partialModels);
    size_t n                     = collection->size();
    TArray<NumericTable *, DAAL_BASE_CPU> partialxtx(n);
    TArray<NumericTable *, DAAL_BASE_CPU> partialxty(n);
    for (size_t i = 0; i < n; i++)
    {
        ridge_regression::ModelNormEq * m = static_cast<ridge_regression::ModelNormEq *>((*collection)[i].get());
        partialxtx[i]                     = m->getXTXTable().get();
        partialxty[i]                     = m->getXTYTable().get();
    }

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(partialResult->get(training::partialModel));

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, n,
                                         partialxtx.get(), partialxty.get(), *(pm->getXTXTable()), *(pm->getXTYTable()));

    collection->clear();
    return s;
}

/**
 *  \brief Choose appropriate kernel to calculate ridge regression model.
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * const partialResult = static_cast<PartialResult *>(_pres);
    Result * const result               = static_cast<Result *>(_res);
    TrainParameter * const par          = static_cast<TrainParameter *>(_par);

    ridge_regression::ModelNormEqPtr pm = ridge_regression::ModelNormEq::cast(partialResult->get(training::partialModel));
    ridge_regression::ModelNormEqPtr m  = ridge_regression::ModelNormEq::cast(result->get(training::model));

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, *(pm->getXTXTable()),
                       *(pm->getXTYTable()), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag, *(par->ridgeParameters));
}

} // namespace internal

} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
