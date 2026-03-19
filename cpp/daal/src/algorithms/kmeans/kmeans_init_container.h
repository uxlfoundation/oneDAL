/* file: kmeans_init_container.h */
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
//  Implementation of K-means algorithm container -- a class that contains
//  Lloyd K-means kernels for supported architectures.
//--
*/

#ifndef __KMEANS_INIT_CONTAINER_H__
#define __KMEANS_INIT_CONTAINER_H__

#include "algorithms/kmeans/kmeans_init_types.h"
#include "algorithms/kmeans/kmeans_init_batch.h"
#include "algorithms/kmeans/kmeans_init_distributed.h"
#include "src/algorithms/kmeans/kmeans_init_kernel.h"
#include "src/algorithms/kmeans/kmeans_init_impl.h"
#include "src/algorithms/engines/engine_factory.h"
namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of initialization of K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::init::Batch class
 *        and supports the method of computing initial clusters for K-Means algorithm in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref daal::algorithms::kmeans::init::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes initial values for K-Means algorithm in the batch processing mode
     */
    services::Status compute() override;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER"></a>
 * \brief Provides methods to run implementations of initialization of K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::init::Distributed class
 *        and supports the method of computing initial clusters for K-Means algorithm in the distributed processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of initial clusters for K-Means algorithm, double or float
 * \tparam method           Method of computing initial clusters for the algorithm, \ref daal::algorithms::kmeans::init::Method
 */
template <ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for computing initial clusters for K-Means algorithm in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means initialization algorithm in the first step of the
     * distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of K-Means initialization algorithm in the first step of the
     * distributed processing mode
     */
    services::Status finalizeCompute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitKernel, method, algorithmFPType);
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

    const size_t na = 1;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());

    const size_t nr = 1;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    Parameter * par = static_cast<Parameter *>(_par);

    daal::services::Environment::env & env = *_env;
    engines::EnginePtr enginePtr = engines::createEngine(par->engine);
    __DAAL_CALL_KERNEL(env, internal::KMeansInitKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par, *enginePtr);
}

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for computing initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for initializing K-Means algorithm with a specified environment
     * in the 2nd step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means initialization algorithm in the 2nd step of the
     * distributed processing mode
     */
    services::Status compute() override;
    /**
     * Computes the result of K-Means initialization algorithm in the 2nd step of the
     * distributed processing mode
     */
    services::Status finalizeCompute() override;
};
/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 2nd step of the distributed processing mode
*        performed on a local node
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 2nd step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 2nd step of the
    * distributed processing mode
    */
    services::Status compute() override;
    /**
    * Computes the result of K-Means initialization algorithm in the 2nd step of the
    * distributed processing mode
    */
    services::Status finalizeCompute() override;
};
/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP3MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 3rd step of the distributed processing mode
*        performed on the master mode
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step3Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 3rd step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 3rd step of the
    * distributed processing mode
    */
    services::Status compute() override;
    /**
    * Computes the result of K-Means initialization algorithm in the 3rd step of the
    * distributed processing mode
    */
    services::Status finalizeCompute() override;
};
/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for comp
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step4Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 4th step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 4th step of the
    * distributed processing mode
    */
    services::Status compute() override;
    /**
    * Computes the result of K-Means initialization algorithm in the 4th step of the
    * distributed processing mode
    */
    services::Status finalizeCompute() override;
};
/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDCONTAINER_STEP5MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
* \brief Class containing methods for computing initial clusters for K-Means algorithm in the 5th step of the distributed processing mode
*        performed on the master node
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step5Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
    * Constructs a container for initializing K-Means algorithm with a specified environment
    * in the 5th step of the distributed processing mode
    * \param[in] daalEnv   Environment object
    */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
    * Computes a partial result of K-Means initialization algorithm in the 5th step of the
    * distributed processing mode
    */
    services::Status compute() override;
    /**
    * Computes the result of K-Means initialization algorithm in the 5th step of the
    * distributed processing mode
    */
    services::Status finalizeCompute() override;
};
template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep1LocalKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    NumericTable * pData = static_cast<Input *>(_in)->get(data).get();
    PartialResult * pRes = static_cast<PartialResult *>(_pres);

    NumericTablePtr pPartialClusters       = pRes->get(partialClusters);
    NumericTable * pNumPartialClusters     = pRes->get(partialClustersNumber).get();
    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;
        engines::EnginePtr enginePtr = engines::createEngine(par->engine);
    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansInitStep1LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
                                                   compute, pData, par, pNumPartialClusters, pPartialClusters, *enginePtr);
    static_cast<PartialResult *>(_pres)->set(partialClusters, pPartialClusters); //can be null
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep2MasterKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    DistributedStep2MasterInput * input       = static_cast<DistributedStep2MasterInput *>(_in);
    Result * result                           = static_cast<Result *>(_res);
    data_management::DataCollection * dcInput = input->get(partialResults).get();

    size_t nPartials = dcInput->size();

    size_t na         = nPartials * 2;
    NumericTable ** a = new NumericTable *[na];
    DAAL_CHECK_MALLOC(a)
    for (size_t i = 0; i < nPartials; i++)
    {
        PartialResult * inPres = static_cast<PartialResult *>((*dcInput)[i].get());
        a[i * 2 + 0]           = static_cast<NumericTable *>(inPres->get(partialClustersNumber).get());
        a[i * 2 + 1]           = static_cast<NumericTable *>(inPres->get(partialClusters).get());
    }

    NumericTable * ntClusters = static_cast<NumericTable *>(result->get(centroids).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansInitStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
                                                   finalizeCompute, na, a, ntClusters, par);

    delete[] a;
    a = nullptr;

    dcInput->clear();
    return s;
}

/////////////////////////////// init plusPlus/parallelPlus distributed containers ///////////////////////////////////////////////
template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep2LocalKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    const DistributedStep2LocalPlusPlusParameter * par    = (const DistributedStep2LocalPlusPlusParameter *)(_par);
    DistributedStep2LocalPlusPlusInput * input            = static_cast<DistributedStep2LocalPlusPlusInput *>(_in);
    const NumericTable * pData                            = input->get(data).get();
    const NumericTable * pNewCenters                      = input->get(inputOfStep2).get();
    DistributedStep2LocalPlusPlusPartialResult * pPartRes = static_cast<DistributedStep2LocalPlusPlusPartialResult *>(_pres);
    NumericTable * pRes                                   = pPartRes->get(outputOfStep2ForStep3).get();
    DataCollectionPtr pLocalData                          = (par->firstIteration ? pPartRes->get(internalResult) : input->get(internalInput));
    NumericTable * aLocalData[internal::localDataSize]    = { 0, 0, 0, 0 };
    for (size_t i = 0; i < pLocalData->size(); ++i) aLocalData[i] = NumericTable::cast((*pLocalData)[i]).get();

    NumericTable * pOutputForStep5 =
        (isParallelPlusMethod(method) && par->outputForStep5Required ? pPartRes->get(outputOfStep2ForStep5).get() : nullptr);

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep2LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, par, pData, pNewCenters,
                       aLocalData, pRes, pOutputForStep5);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep3MasterKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep3MasterPlusPlusInput * input      = static_cast<DistributedStep3MasterPlusPlusInput *>(_in);
    DistributedStep3MasterPlusPlusPartialResult * pr = static_cast<DistributedStep3MasterPlusPlusPartialResult *>(_pres);
    data_management::MemoryBlock * pRngState         = dynamic_cast<data_management::MemoryBlock *>(pr->get(rngState).get());
    DAAL_CHECK(pRngState, services::ErrorNullPtr);
    const Parameter * par = (const Parameter *)(_par);
        engines::EnginePtr enginePtr = engines::createEngine(par->engine);
    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep3MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, par,
                       input->get(inputOfStep3FromStep2).get(), pRngState, pr->get(outputOfStep3ForStep4).get(), *enginePtr);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep4LocalKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::compute()
{
    DistributedStep4LocalPlusPlusInput * input            = static_cast<DistributedStep4LocalPlusPlusInput *>(_in);
    const NumericTable * pData                            = input->get(data).get();
    const NumericTable * pInput                           = input->get(inputOfStep4FromStep3).get();
    DistributedStep4LocalPlusPlusPartialResult * pPartRes = static_cast<DistributedStep4LocalPlusPlusPartialResult *>(_pres);
    NumericTable * pOutput                                = pPartRes->get(outputOfStep4).get();
    DataCollectionPtr pLocalData                          = input->get(internalInput);
    NumericTable * aLocalData[internal::localDataSize]    = { 0, 0, 0, 0 };
    for (size_t i = 0; i < pLocalData->size(); ++i) aLocalData[i] = NumericTable::cast((*pLocalData)[i]).get();

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep4LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, pData, pInput,
                       aLocalData, pOutput);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep5MasterKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep5MasterPlusPlusInput * input            = static_cast<DistributedStep5MasterPlusPlusInput *>(_in);
    const DataCollection * pCandidates                     = input->get(inputCentroids).get();
    const DataCollection * pRating                         = input->get(inputOfStep5FromStep2).get();
    DistributedStep5MasterPlusPlusPartialResult * pPartRes = static_cast<DistributedStep5MasterPlusPlusPartialResult *>(_pres);
    NumericTable * pResCand                                = pPartRes->get(candidates).get();
    NumericTable * pResWeights                             = pPartRes->get(weights).get();
    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep5MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, pCandidates, pRating,
                       pResCand, pResWeights);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    const Parameter * par                                  = (const Parameter *)(_par);
    DistributedStep5MasterPlusPlusPartialResult * pPartRes = static_cast<DistributedStep5MasterPlusPlusPartialResult *>(_pres);
    DistributedStep5MasterPlusPlusInput * input            = static_cast<DistributedStep5MasterPlusPlusInput *>(_in);
    NumericTable * ntCandidates                            = pPartRes->get(candidates).get();
    NumericTable * ntWeights                               = pPartRes->get(weights).get();
    data_management::MemoryBlock * pRngState               = dynamic_cast<data_management::MemoryBlock *>(input->get(inputOfStep5FromStep3).get());
    Result * pRes                                          = static_cast<Result *>(_res);
    DAAL_CHECK(pRngState, services::ErrorNullPtr);
    engines::EnginePtr enginePtr = engines::createEngine(par->engine);
    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep5MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, par,
                       ntCandidates, ntWeights, pRngState, pRes->get(centroids).get(), *enginePtr);
}

} // namespace internal

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
