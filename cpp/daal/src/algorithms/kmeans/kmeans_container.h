/* file: kmeans_container.h */
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

#ifndef __KMEANS_CONTAINER_H__
#define __KMEANS_CONTAINER_H__

#include "algorithms/kmeans/kmeans_types.h"
#include "algorithms/kmeans/kmeans_batch.h"
#include "algorithms/kmeans/kmeans_distributed.h"
#include "src/algorithms/kmeans/kmeans_lloyd_kernel.h"

#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of K-Means algorithm.
 *        This class is associated with the daal::algorithms::kmeans::Batch class
 *        and supports the method of K-Means computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of K-Means, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::kmeans::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for K-Means algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of K-Means algorithm in the batch processing mode
     */
    virtual services::Status compute() override;
};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for K-Means algorithm in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step1Local, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for K-Means algorithm with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status compute() override;
    /**
     * Computes the result of K-Means algorithm in the first step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansBatchKernel, method, algorithmFPType);
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

    NumericTable * a[lastInputId + 1] = { input->get(data).get(), input->get(inputCentroids).get() };

    NumericTable * r[lastResultId + 1] = { result->get(centroids).get(), result->get(assignments).get(), result->get(objectiveFunction).get(),
                                           result->get(nIterations).get() };

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansBatchKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a, r, par);
}


/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for K-Means algorithm in the second step of the distributed processing mode
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DistributedContainer<step2Master, algorithmFPType, method, cpu> : public daal::algorithms::AnalysisContainerIface<distributed>
{
public:
    /**
     * Constructs a container for K-Means algorithm with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~DistributedContainer();
    /**
     * Computes a partial result of K-Means algorithm in the second step of the
     * distributed processing mode
     */
    virtual services::Status compute() override;
    /**
     * Computes the result of K-Means algorithm in the second step of the
     * distributed processing mode
     */
    virtual services::Status finalizeCompute() override;
};
template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep1Kernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    Input * input        = static_cast<Input *>(_in);
    PartialResult * pres = static_cast<PartialResult *>(_pres);
    Parameter * par      = static_cast<Parameter *>(_par);

    const size_t na = 2;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());
    a[1] = static_cast<NumericTable *>(input->get(inputCentroids).get());

    const size_t isAssignments = par->resultsToEvaluate & computeAssignments || par->assignFlag;
    const size_t nr            = 5 + isAssignments;
    NumericTable * r[6];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());
    if (isAssignments)
    {
        r[5] = static_cast<NumericTable *>(pres->get(partialAssignments).get());
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r, par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * pres = static_cast<PartialResult *>(_pres);
    Result * res         = static_cast<Result *>(_res);
    Parameter * par      = static_cast<Parameter *>(_par);

    const size_t na = 1;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(pres->get(partialAssignments).get());

    const size_t nr = 1;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(res->get(assignments).get());

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep1Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r,
                       par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansDistributedStep2Kernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep2MasterInput * input       = static_cast<DistributedStep2MasterInput *>(_in);
    data_management::DataCollection * dcInput = input->get(partialResults).get();
    PartialResult * pres                      = static_cast<PartialResult *>(_pres);

    size_t nPartials = dcInput->size();

    size_t na = nPartials * 5;
    daal::internal::TArray<NumericTable *, cpu> aPtr(na);
    NumericTable ** a = aPtr.get();
    DAAL_CHECK_MALLOC(a);
    for (size_t i = 0; i < nPartials; i++)
    {
        PartialResult * inPres = static_cast<PartialResult *>((*dcInput)[i].get());
        a[i * 5 + 0]           = static_cast<NumericTable *>(inPres->get(nObservations).get());
        a[i * 5 + 1]           = static_cast<NumericTable *>(inPres->get(partialSums).get());
        a[i * 5 + 2]           = static_cast<NumericTable *>(inPres->get(partialObjectiveFunction).get());
        a[i * 5 + 3]           = static_cast<NumericTable *>(inPres->get(partialCandidatesDistances).get());
        a[i * 5 + 4]           = static_cast<NumericTable *>(inPres->get(partialCandidatesCentroids).get());
    }

    const size_t nr = 5;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    r[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    r[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    r[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    r[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    services::Status s;

    s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr,
                                  r, par);
    dcInput->clear();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * pres = static_cast<PartialResult *>(_pres);
    Result * result      = static_cast<Result *>(_res);

    const size_t na = 5;
    NumericTable * a[na];
    a[0] = static_cast<NumericTable *>(pres->get(nObservations).get());
    a[1] = static_cast<NumericTable *>(pres->get(partialSums).get());
    a[2] = static_cast<NumericTable *>(pres->get(partialObjectiveFunction).get());
    a[3] = static_cast<NumericTable *>(pres->get(partialCandidatesDistances).get());
    a[4] = static_cast<NumericTable *>(pres->get(partialCandidatesCentroids).get());

    const size_t nr = 2;
    NumericTable * r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());
    r[1] = static_cast<NumericTable *>(result->get(objectiveFunction).get());

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansDistributedStep2Kernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute, na, a, nr, r,
                       par);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
