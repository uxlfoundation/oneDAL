/* file: linear_regression_train_container.h */
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
//  Implementation of linear regression container.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_CONTAINER_H__
#define __LINEAR_REGRESSION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/linear_regression/linear_regression_training_batch.h"
#include "algorithms/linear_regression/linear_regression_training_online.h"
#include "algorithms/linear_regression/linear_regression_training_distributed.h"
#include "src/algorithms/linear_regression/linear_regression_train_kernel.h"
#include "algorithms/linear_regression/linear_regression_ne_model.h"
#include "algorithms/linear_regression/linear_regression_qr_model.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;

/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods for normal equations linear regression
 *        model-based training using algorithmFPType precision arithmetic
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for linear regression model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of linear regression model-based training in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__ONLINECONTAINER"></a>
 * \brief Class containing methods for linear regression model-based training
 * in the online processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public TrainingContainerIface<online>
{
public:
    /**
     * Constructs a container for linear regression model-based training with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~OnlineContainer();

    /**
     * Computes a partial result of linear regression model-based training
     * in the online processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
    /**
     * Computes the result of linear regression model-based training
     * in the online processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods for linear regression model-based training in the distributed processing mode
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for linear regression model-based training
 * in the second step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for linear regression model-based training with a specified environment
     * in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes a partial result of linear regression model-based training
     * in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status compute() override;
    /**
     * Computes the result of linear regression model-based training
     * in the second step of the distributed processing mode
     *
     * \return Status of computations
     */
    services::Status finalizeCompute() override;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);
    Parameter * par = static_cast<Parameter *>(_par);

    Environment::env & env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr m = linear_regression::ModelNormEq::cast(result->get(model));

        __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), compute, *(input->get(data)),
                           *(input->get(dependentVariables)), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr m = linear_regression::ModelQR::cast(result->get(model));

        __DAAL_CALL_KERNEL(env, internal::BatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense), compute, *(input->get(data)),
                           *(input->get(dependentVariables)), *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}

/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    linear_regression::training::Input * input = static_cast<linear_regression::training::Input *>(_in);
    PartialResult * partialResult              = static_cast<PartialResult *>(_pres);
    Parameter * par                            = static_cast<Parameter *>(_par);

    Environment::env & env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr m = linear_regression::ModelNormEq::cast(partialResult->get(training::partialModel));

        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), compute, *(input->get(data)),
                           *(input->get(dependentVariables)), *(m->getXTXTable()), *(m->getXTYTable()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr m = linear_regression::ModelQR::cast(partialResult->get(training::partialModel));

        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense), compute, *(input->get(data)),
                           *(input->get(dependentVariables)), *(m->getRTable()), *(m->getQTYTable()), par->interceptFlag);
    }
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);
    Parameter * par               = static_cast<Parameter *>(_par);
    Environment::env & env        = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(partialResult->get(training::partialModel));
        linear_regression::ModelNormEqPtr m  = linear_regression::ModelNormEq::cast(result->get(training::model));

        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), finalizeCompute,
                           *(pm->getXTXTable()), *(pm->getXTYTable()), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(partialResult->get(training::partialModel));
        linear_regression::ModelQRPtr m  = linear_regression::ModelQR::cast(result->get(training::model));
        __DAAL_CALL_KERNEL(env, internal::OnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense), finalizeCompute,
                           *(pm->getRTable()), *(pm->getQTYTable()), *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}

/**
 *  \brief Initialize list of linear regression
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::DistributedKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> * input = static_cast<DistributedInput<step2Master> *>(_in);
    PartialResult * partialResult         = static_cast<PartialResult *>(_pres);

    Environment::env & env = *_env;

    DataCollectionPtr collection = input->get(partialModels);
    size_t n                     = collection->size();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(NumericTable *));

    Status s;
    if (method == training::normEqDense)
    {
        TArray<NumericTable *, DAAL_BASE_CPU> partialxtx(n);
        TArray<NumericTable *, DAAL_BASE_CPU> partialxty(n);
        for (size_t i = 0; i < n; i++)
        {
            linear_regression::ModelNormEq * m = static_cast<linear_regression::ModelNormEq *>((*collection)[i].get());
            partialxtx[i]                      = m->getXTXTable().get();
            partialxty[i]                      = m->getXTYTable().get();
        }

        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(partialResult->get(training::partialModel));

        s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), compute, n,
                                      partialxtx.get(), partialxty.get(), *(pm->getXTXTable()), *(pm->getXTYTable()));
    }
    else
    {
        TArray<NumericTable *, DAAL_BASE_CPU> partialr(n);
        TArray<NumericTable *, DAAL_BASE_CPU> partialqty(n);
        for (size_t i = 0; i < n; i++)
        {
            linear_regression::ModelQR * m = static_cast<linear_regression::ModelQR *>((*collection)[i].get());
            partialr[i]                    = m->getRTable().get();
            partialqty[i]                  = m->getQTYTable().get();
        }

        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(partialResult->get(training::partialModel));

        s = __DAAL_CALL_KERNEL_STATUS(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense), compute, n,
                                      partialr.get(), partialqty.get(), *(pm->getRTable()), *(pm->getQTYTable()));
    }
    collection->clear();
    return s;
}

/**
 *  \brief Choose appropriate kernel to calculate linear regression model.
 *
 *  \param env[in]  Environment
 *  \param a[in]    Array of numeric tables contating input data
 *  \param r[out]   Resulting model
 *  \param par[in]  Linear regression algorithm parameters
 */
template <typename algorithmFPType, training::Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);
    Parameter * par               = static_cast<Parameter *>(_par);

    Environment::env & env = *_env;

    if (method == training::normEqDense)
    {
        linear_regression::ModelNormEqPtr pm = linear_regression::ModelNormEq::cast(partialResult->get(training::partialModel));
        linear_regression::ModelNormEqPtr m  = linear_regression::ModelNormEq::cast(result->get(training::model));
        __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::normEqDense), finalizeCompute,
                           *(pm->getXTXTable()), *(pm->getXTYTable()), *(m->getXTXTable()), *(m->getXTYTable()), *(m->getBeta()), par->interceptFlag);
    }
    else
    {
        linear_regression::ModelQRPtr pm = linear_regression::ModelQR::cast(partialResult->get(training::partialModel));
        linear_regression::ModelQRPtr m  = linear_regression::ModelQR::cast(result->get(training::model));

        __DAAL_CALL_KERNEL(env, internal::DistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, training::qrDense), finalizeCompute,
                           *(pm->getRTable()), *(pm->getQTYTable()), *(m->getRTable()), *(m->getQTYTable()), *(m->getBeta()), par->interceptFlag);
    }
}

} // namespace internal

} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
