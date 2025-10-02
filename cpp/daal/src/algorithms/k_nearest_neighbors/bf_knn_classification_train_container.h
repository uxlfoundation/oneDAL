/* file: bf_knn_classification_train_container.h */
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

#ifndef __BF_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __BF_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_shared_ptr.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_batch.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{
using namespace daal::data_management;

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainKernel, algorithmFpType);
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    services::Status status;
    const bf_knn_classification::Parameter * const par         = static_cast<bf_knn_classification::Parameter *>(_par);
    const bf_knn_classification::training::Input * const input = static_cast<bf_knn_classification::training::Input *>(_in);
    Result * const result                                      = static_cast<Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);

    const bf_knn_classification::ModelPtr r = result->get(classifier::training::model);

    daal::services::Environment::env & env = *_env;

    const bool copy = (par->dataUseInModel == doNotUse);
    status |= r->impl()->setData<algorithmFpType>(x, copy);
    DAAL_CHECK_STATUS_VAR(status);
    if ((par->resultsToEvaluate & daal::algorithms::classifier::computeClassLabels) != 0)
    {
        const NumericTablePtr y = input->get(classifier::training::labels);
        status |= r->impl()->setLabels<algorithmFpType>(y, copy);
    }
    DAAL_CHECK_STATUS_VAR(status);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute, r->impl()->getData().get(),
                       r->impl()->getLabels().get(), r.get(), *par, *par->engine);
}

} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
