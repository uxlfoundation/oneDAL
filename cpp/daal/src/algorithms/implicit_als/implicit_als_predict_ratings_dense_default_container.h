/* file: implicit_als_predict_ratings_dense_default_container.h */
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
//  Implementation of implicit ALS prediction algorithm container.
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_CONTAINER_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_CONTAINER_H__

#include "algorithms/implicit_als/implicit_als_predict_ratings_batch.h"
#include "src/algorithms/implicit_als/implicit_als_predict_ratings_dense_default_kernel.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace internal
{
/**
 *  \brief Initialize list of implicit ALS prediction algorithm
 *  kernels with implementations for supported architectures
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the implicit ALS ratings prediction algorithm in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for implicit ALS model-based prediction, double or float
 * \tparam method           Implicit ALS prediction method, \ref Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for implicit ALS model-based ratings prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of implicit ALS model-based ratings prediction
     * in the batch processing mode
     */
    services::Status compute() override;
};

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSPredictKernel, algorithmFPType);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    Model * alsModel = static_cast<Model *>(input->get(model).get());

    NumericTable * usersFactorsTable = alsModel->getUsersFactors().get();
    NumericTable * itemsFactorsTable = alsModel->getItemsFactors().get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    NumericTable * ratingsTable = static_cast<NumericTable *>(result->get(prediction).get());
    __DAAL_CALL_KERNEL(env, internal::ImplicitALSPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, usersFactorsTable,
                       itemsFactorsTable, ratingsTable, par);
}

/**
 *  \brief Initialize list of implicit ALS prediction algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
    : DistributedPredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSPredictKernel, algorithmFPType);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> * input = static_cast<DistributedInput<step1Local> *>(_in);
    PartialResult * partialResult        = static_cast<PartialResult *>(_pres);
    Result * result                      = static_cast<Result *>(partialResult->get(finalResult).get());

    PartialModel * usersFactors = static_cast<PartialModel *>(input->get(usersPartialModel).get());
    PartialModel * itemsFactors = static_cast<PartialModel *>(input->get(itemsPartialModel).get());

    NumericTablePtr usersFactorsTable = usersFactors->getFactors();
    NumericTablePtr itemsFactorsTable = itemsFactors->getFactors();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    NumericTable * ratingsTable = static_cast<NumericTable *>(result->get(prediction).get());

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), compute, usersFactorsTable.get(),
                       itemsFactorsTable.get(), ratingsTable, par);
}

template <typename algorithmFPType, prediction::ratings::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // namespace internal

} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
