/* file: svd_dense_default_container.h */
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
//  Implementation of svd calculation algorithm container.
//--
*/

//#include "svd.h"
#include "algorithms/svd/svd_types.h"
#include "algorithms/svd/svd_batch.h"
#include "algorithms/svd/svd_online.h"
#include "algorithms/svd/svd_distributed.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"
#include "src/algorithms/svd/svd_dense_default_kernel.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
using namespace daal::internal;

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the SVD algorithm in the batch processing mode
     */
    services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__ONLINECONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm in the online processing mode
 *
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class OnlineContainer : public daal::algorithms::AnalysisContainerIface<online>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the online processing mode
     * \param[in] daalEnv   Environment object
     */
    OnlineContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~OnlineContainer();
    /**
     * Computes a partial result of the SVD algorithm in the online processing mode
     * \return Status of computations
     */
    services::Status compute() override;
    /**
     * Computes the result of the SVD algorithm in the online processing mode
     * \return Status of computations
     */
    services::Status finalizeCompute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the SVD algorithm.
 *
 * \tparam step             Step of the computing algorithm in the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           Computation method, \ref daal::algorithms::svd::Method
 *
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the second step of the SVD algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the SVD algorithm in the second step
     * of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the SVD algorithm in the second step
     * of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of the third step of the SVD algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVD algorithm, double or float
 * \tparam method           SVD computation method, \ref daal::algorithms::svd::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for the SVD algorithm with a specified environment
     * in the third step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of the SVD algorithm in the third step
     * of the distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of the SVD algorithm in the third step
     * of the distributed processing mode
     */
    services::Status finalizeCompute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDBatchKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    const size_t na = input->size();
    const size_t nr = result->size();

    NumericTable * a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable ** a = &a0;
    NumericTable * r[3];
    r[0]                                   = static_cast<NumericTable *>(result->get(singularValues).get());
    r[1]                                   = static_cast<NumericTable *>(result->get(leftSingularMatrix).get());
    r[2]                                   = static_cast<NumericTable *>(result->get(rightSingularMatrix).get());
    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDOnlineKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input                       = static_cast<Input *>(_in);
    OnlinePartialResult * partialResult = static_cast<OnlinePartialResult *>(_pres);
    Parameter & svdPar                  = *(static_cast<Parameter *>(_par));

    const size_t na = input->size();

    NumericTable * a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable ** a = &a0;

    size_t m = a0->getNumberOfColumns();
    size_t n = a0->getNumberOfRows();

    Status s = partialResult->addPartialResultStorage<algorithmFPType>(m, n, svdPar);
    DAAL_CHECK_STATUS_VAR(s)

    const size_t nr                               = 2;
    data_management::DataCollection * rCollection = static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep2).get());
    size_t np                                     = rCollection->size();

    NumericTable * r[2] = { 0, 0 };

    if (svdPar.leftSingularMatrix != notRequired)
    {
        data_management::DataCollection * qCollection =
            static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep3).get());
        r[0] = static_cast<NumericTable *>((*qCollection)[np - 1].get());
    }
    r[1] = static_cast<NumericTable *>((*rCollection)[np - 1].get());

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, &svdPar);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    Parameter & svdPar                  = *(static_cast<Parameter *>(_par));
    OnlinePartialResult * partialResult = static_cast<OnlinePartialResult *>(_pres);
    Result * result                     = static_cast<Result *>(_res);

    data_management::DataCollection * qCollection = static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep3).get());
    data_management::DataCollection * rCollection = static_cast<data_management::DataCollection *>(partialResult->get(outputOfStep1ForStep2).get());
    size_t np                                     = rCollection->size();

    daal::internal::TArray<NumericTable *, cpu> aPtr(np * 2);
    NumericTable ** a = aPtr.get();

    for (size_t i = 0; i < np; i++)
    {
        a[i] = static_cast<NumericTable *>((*rCollection)[i].get());
        if (svdPar.leftSingularMatrix != notRequired)
        {
            a[i + np] = static_cast<NumericTable *>((*qCollection)[i].get());
        }
        else
        {
            a[i + np] = 0;
        }
    }
    const size_t na = np * 2;

    const size_t nr = 3;
    NumericTable * r[3];
    r[0] = static_cast<NumericTable *>(result->get(singularValues).get());
    r[1] = static_cast<NumericTable *>(result->get(leftSingularMatrix).get());
    r[2] = static_cast<NumericTable *>(result->get(rightSingularMatrix).get());

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute, na, a, nr, r, &svdPar);
}

/**
 *  \brief Initialize list of cholesky kernels with implementations for supported architectures
 */
template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDDistributedStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2Input * input            = static_cast<DistributedStep2Input *>(_in);
    DistributedPartialResult * partialResult = static_cast<DistributedPartialResult *>(_pres);

    data_management::KeyValueDataCollection * inCollection =
        static_cast<data_management::KeyValueDataCollection *>(input->get(inputOfStep2FromStep1).get());

    size_t nBlocks = input->getNBlocks();
    size_t nNodes  = inCollection->size();

    data_management::KeyValueDataCollection * perNodePartials =
        static_cast<data_management::KeyValueDataCollection *>(partialResult->get(outputOfStep2ForStep3).get());
    Result * results = static_cast<Result *>(partialResult->get(finalResultFromStep2Master).get());

    const size_t na = nBlocks;

    daal::internal::TArray<NumericTable *, cpu> aPtr(nBlocks);
    NumericTable ** a = aPtr.get();

    const size_t nr = nBlocks + 2;
    daal::internal::TArray<NumericTable *, cpu> rPtr(nBlocks + 2);
    NumericTable ** r = rPtr.get();

    r[0] = static_cast<NumericTable *>(results->get(singularValues).get());
    r[1] = static_cast<NumericTable *>(results->get(rightSingularMatrix).get());

    size_t iBlocks = 0;
    for (size_t i = 0; i < nNodes; i++)
    {
        data_management::DataCollection * nodeCollection = static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex(i).get());
        data_management::DataCollection * nodePartials = static_cast<data_management::DataCollection *>((*perNodePartials).getValueByIndex(i).get());

        size_t nodeSize = nodeCollection->size();

        for (size_t j = 0; j < nodeSize; j++)
        {
            a[iBlocks + j]     = static_cast<NumericTable *>((*nodeCollection)[j].get());
            r[iBlocks + j + 2] = static_cast<NumericTable *>((*nodePartials)[j].get());
        }

        iBlocks += nodeSize;
    }

    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::SVDDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a,
                                         nr, r, par);

    inCollection->clear();

    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVDDistributedStep3Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedStep3Input * input                 = static_cast<DistributedStep3Input *>(_in);
    DistributedPartialResultStep3 * partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);

    data_management::DataCollectionPtr qCollection = input->get(inputOfStep3FromStep1);
    data_management::DataCollectionPtr rCollection = input->get(inputOfStep3FromStep2);

    ResultPtr result = partialResult->get(finalResultFromStep3);

    size_t nBlocks = qCollection->size();

    const size_t na = nBlocks * 2;

    daal::internal::TArray<NumericTable *, cpu> aPtr(na);
    NumericTable ** a = aPtr.get();

    for (size_t i = 0; i < nBlocks; i++)
    {
        a[i]           = static_cast<NumericTable *>((*qCollection)[i].get());
        a[i + nBlocks] = static_cast<NumericTable *>((*rCollection)[i].get());
    }

    size_t nr = 1;
    NumericTable * r[1];
    r[0] = static_cast<NumericTable *>(result->get(leftSingularMatrix).get());

    daal::algorithms::Parameter * par      = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVDDistributedStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, na, a, nr, r, par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return Status();
}

} // namespace internal

} // namespace svd
} // namespace algorithms
} // namespace daal
