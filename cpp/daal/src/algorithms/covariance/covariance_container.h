/* file: covariance_container.h */
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
//  Implementation of Covariance algorithm container.
//--
*/

#ifndef __COVARIANCE_CONTAINER_H__
#define __COVARIANCE_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/covariance/covariance_batch.h"
#include "algorithms/covariance/covariance_online.h"
#include "algorithms/covariance/covariance_distributed.h"
#include "src/algorithms/covariance/covariance_hyperparameter_impl.h"
#include "src/algorithms/covariance/covariance_kernel.h"

#undef __DAAL_CONCAT
#define __DAAL_CONCAT(x, y) x##y

#undef __DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR
#define __DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(ComputeMethod, KernelClass)                                   \
    template <typename algorithmFPType, CpuType cpu>                                                                \
    BatchContainer<algorithmFPType, ComputeMethod, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) \
    {                                                                                                               \
        __DAAL_INITIALIZE_KERNELS(KernelClass, algorithmFPType, ComputeMethod);                                     \
    }

#undef __DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR
#define __DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(ComputeMethod)        \
    template <typename algorithmFPType, CpuType cpu>                       \
    BatchContainer<algorithmFPType, ComputeMethod, cpu>::~BatchContainer() \
    {                                                                      \
        __DAAL_DEINITIALIZE_KERNELS();                                     \
    }

#undef __DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE
#define __DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(ComputeMethod, KernelClass)                                                                  \
    template <typename algorithmFPType, CpuType cpu>                                                                                           \
    services::Status BatchContainer<algorithmFPType, ComputeMethod, cpu>::compute()                                                            \
    {                                                                                                                                          \
        Result * result = static_cast<Result *>(_res);                                                                                         \
        Input * input   = static_cast<Input *>(_in);                                                                                           \
                                                                                                                                               \
        NumericTable * dataTable = input->get(data).get();                                                                                     \
        NumericTable * covTable  = result->get(covariance).get();                                                                              \
        NumericTable * meanTable = result->get(mean).get();                                                                                    \
                                                                                                                                               \
        Parameter * parameter                           = static_cast<Parameter *>(_par);                                                      \
        const internal::Hyperparameter * hyperparameter = static_cast<const internal::Hyperparameter *>(_hpar);                                \
        daal::services::Environment::env & env          = *_env;                                                                               \
                                                                                                                                               \
        __DAAL_CALL_KERNEL(env, KernelClass, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod), compute, dataTable, covTable, meanTable, \
                           parameter, hyperparameter);                                                                                         \
    }

#undef __DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(ComputeMethod, KernelClass)                                    \
    template <typename algorithmFPType, CpuType cpu>                                                                  \
    OnlineContainer<algorithmFPType, ComputeMethod, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv) \
    {                                                                                                                 \
        __DAAL_INITIALIZE_KERNELS(KernelClass, algorithmFPType, ComputeMethod);                                       \
    }

#undef __DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(ComputeMethod)         \
    template <typename algorithmFPType, CpuType cpu>                         \
    OnlineContainer<algorithmFPType, ComputeMethod, cpu>::~OnlineContainer() \
    {                                                                        \
        __DAAL_DEINITIALIZE_KERNELS();                                       \
    }

#undef __DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(ComputeMethod, KernelClass)                                                       \
    template <typename algorithmFPType, CpuType cpu>                                                                                 \
    services::Status OnlineContainer<algorithmFPType, ComputeMethod, cpu>::compute()                                                 \
    {                                                                                                                                \
        PartialResult * partialResult = static_cast<PartialResult *>(_pres);                                                         \
        Input * input                 = static_cast<Input *>(_in);                                                                   \
                                                                                                                                     \
        NumericTable * dataTable = input->get(data).get();                                                                           \
                                                                                                                                     \
        NumericTable * nObsTable         = partialResult->get(nObservations).get();                                                  \
        NumericTable * crossProductTable = partialResult->get(crossProduct).get();                                                   \
        NumericTable * sumTable          = partialResult->get(sum).get();                                                            \
                                                                                                                                     \
        Parameter * parameter                           = static_cast<Parameter *>(_par);                                            \
        const internal::Hyperparameter * hyperparameter = static_cast<const internal::Hyperparameter *>(_hpar);                      \
        daal::services::Environment::env & env          = *_env;                                                                     \
                                                                                                                                     \
        __DAAL_CALL_KERNEL(env, KernelClass, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod), compute, dataTable, nObsTable, \
                           crossProductTable, sumTable, parameter, hyperparameter);                                                  \
    }

#undef __DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE
#define __DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(ComputeMethod, KernelClass)                                                               \
    template <typename algorithmFPType, CpuType cpu>                                                                                                 \
    services::Status OnlineContainer<algorithmFPType, ComputeMethod, cpu>::finalizeCompute()                                                         \
    {                                                                                                                                                \
        PartialResult * partialResult = static_cast<PartialResult *>(_pres);                                                                         \
        Result * result               = static_cast<Result *>(_res);                                                                                 \
                                                                                                                                                     \
        NumericTable * nObsTable         = partialResult->get(nObservations).get();                                                                  \
        NumericTable * crossProductTable = partialResult->get(crossProduct).get();                                                                   \
        NumericTable * sumTable          = partialResult->get(sum).get();                                                                            \
                                                                                                                                                     \
        NumericTable * covTable  = result->get(covariance).get();                                                                                    \
        NumericTable * meanTable = result->get(mean).get();                                                                                          \
                                                                                                                                                     \
        Parameter * parameter                           = static_cast<Parameter *>(_par);                                                            \
        const internal::Hyperparameter * hyperparameter = static_cast<const internal::Hyperparameter *>(_hpar);                                      \
        daal::services::Environment::env & env          = *_env;                                                                                     \
                                                                                                                                                     \
        __DAAL_CALL_KERNEL(env, KernelClass, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod), finalizeCompute, nObsTable, crossProductTable, \
                           sumTable, covTable, meanTable, parameter, hyperparameter);                                                                \
    }

#undef __DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR
#define __DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(ComputeMethod)                                                                         \
    template <typename algorithmFPType, CpuType cpu>                                                                                         \
    DistributedContainer<step2Master, algorithmFPType, ComputeMethod, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv) \
    {                                                                                                                                        \
        __DAAL_INITIALIZE_KERNELS(internal::CovarianceDistributedKernel, algorithmFPType, ComputeMethod);                                    \
    }

#undef __DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR
#define __DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(ComputeMethod)                                 \
    template <typename algorithmFPType, CpuType cpu>                                                \
    DistributedContainer<step2Master, algorithmFPType, ComputeMethod, cpu>::~DistributedContainer() \
    {                                                                                               \
        __DAAL_DEINITIALIZE_KERNELS();                                                              \
    }

#undef __DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE
#define __DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(ComputeMethod)                                                                                     \
    template <typename algorithmFPType, CpuType cpu>                                                                                                 \
    services::Status DistributedContainer<step2Master, algorithmFPType, ComputeMethod, cpu>::compute()                                               \
    {                                                                                                                                                \
        PartialResult * partialResult                   = static_cast<PartialResult *>(_pres);                                                       \
        DistributedInput<step2Master> * input           = static_cast<DistributedInput<step2Master> *>(_in);                                         \
        DataCollection * collection                     = input->get(partialResults).get();                                                          \
        NumericTable * nObsTable                        = partialResult->get(nObservations).get();                                                   \
        NumericTable * crossProductTable                = partialResult->get(crossProduct).get();                                                    \
        NumericTable * sumTable                         = partialResult->get(sum).get();                                                             \
        Parameter * parameter                           = static_cast<Parameter *>(_par);                                                            \
        const internal::Hyperparameter * hyperparameter = static_cast<const internal::Hyperparameter *>(_hpar);                                      \
        daal::services::Environment::env & env          = *_env;                                                                                     \
                                                                                                                                                     \
        __DAAL_CALL_KERNEL(env, internal::CovarianceDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod), compute, collection, \
                           nObsTable, crossProductTable, sumTable, parameter, hyperparameter);                                                       \
                                                                                                                                                     \
        collection->clear();                                                                                                                         \
    }

#undef __DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE
#define __DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(ComputeMethod)                                                                         \
    template <typename algorithmFPType, CpuType cpu>                                                                                             \
    services::Status DistributedContainer<step2Master, algorithmFPType, ComputeMethod, cpu>::finalizeCompute()                                   \
    {                                                                                                                                            \
        Result * result                                 = static_cast<Result *>(_res);                                                           \
        PartialResult * partialResult                   = static_cast<PartialResult *>(_pres);                                                   \
        NumericTable * nObsTable                        = partialResult->get(nObservations).get();                                               \
        NumericTable * crossProductTable                = partialResult->get(crossProduct).get();                                                \
        NumericTable * sumTable                         = partialResult->get(sum).get();                                                         \
        NumericTable * covTable                         = result->get(covariance).get();                                                         \
        NumericTable * meanTable                        = result->get(mean).get();                                                               \
        Parameter * parameter                           = static_cast<Parameter *>(_par);                                                        \
        const internal::Hyperparameter * hyperparameter = static_cast<const internal::Hyperparameter *>(_hpar);                                  \
        daal::services::Environment::env & env          = *_env;                                                                                 \
                                                                                                                                                 \
        __DAAL_CALL_KERNEL(env, internal::CovarianceDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, ComputeMethod), finalizeCompute, \
                           nObsTable, crossProductTable, sumTable, covTable, meanTable, parameter, hyperparameter);                              \
    }

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm.
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::covariance::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_DEFAULTDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using default computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, defaultDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SINGLEPASSDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, singlePassDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SUMDENSE_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, sumDense, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_FASTCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using fast computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, fastCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SINGLEPASSCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using single-pass computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, singlePassCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCHCONTAINER_ALGORITHMFPTYPE_SUMCSR_CPU"></a>
 * \brief Provides methods to run implementations of the correlation or variance-covariance matrix algorithm
 *        using precomputed sum computation method that works with Compressed Sparse Rows (CSR) numeric tables
 *        This class is associated with daal::algorithms::covariance::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 */
template <typename algorithmFPType, CpuType cpu>
class BatchContainer<algorithmFPType, sumCSR, cpu> : public BatchContainerIface
{
public:
    /**
     * Constructs a container for the correlation or variance-covariance matrix algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();

    /**
     * Computes the result of the correlation or variance-covariance matrix algorithm
     * in the batch processing mode
     */
    virtual services::Status compute() override;
};

__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(defaultDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(singlePassDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(sumDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(fastCSR, internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(singlePassCSR, internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_CONSTRUCTOR(sumCSR, internal::CovarianceCSRBatchKernel)

__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(defaultDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(sumDense)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(fastCSR)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_BATCH_CONTAINER_DESTRUCTOR(sumCSR)

__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(defaultDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(singlePassDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(sumDense, internal::CovarianceDenseBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(fastCSR, internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(singlePassCSR, internal::CovarianceCSRBatchKernel)
__DAAL_COVARIANCE_BATCH_CONTAINER_COMPUTE(sumCSR, internal::CovarianceCSRBatchKernel)

} // namespace internal

__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(defaultDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(sumDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(fastCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(singlePassCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_CONSTRUCTOR(sumCSR, internal::CovarianceCSROnlineKernel)

__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(defaultDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(sumDense)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(fastCSR)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_ONLINE_CONTAINER_DESTRUCTOR(sumCSR)

__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(defaultDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(sumDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(fastCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(singlePassCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_COMPUTE(sumCSR, internal::CovarianceCSROnlineKernel)

__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(defaultDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(singlePassDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(sumDense, internal::CovarianceDenseOnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(fastCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(singlePassCSR, internal::CovarianceCSROnlineKernel)
__DAAL_COVARIANCE_ONLINE_CONTAINER_FINALIZECOMPUTE(sumCSR, internal::CovarianceCSROnlineKernel)

__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(defaultDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(sumDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(fastCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_CONSTRUCTOR(sumCSR)

__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(defaultDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(singlePassDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(sumDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(fastCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(singlePassCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_DESTRUCTOR(sumCSR)

__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(defaultDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(singlePassDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(sumDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(fastCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(singlePassCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_COMPUTE(sumCSR)

__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(defaultDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(singlePassDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(sumDense)
__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(fastCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(singlePassCSR)
__DAAL_COVARIANCE_DISTR_CONTAINER_FINALIZECOMPUTE(sumCSR)

} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
