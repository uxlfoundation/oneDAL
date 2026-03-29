/* file: kdtree_knn_classification_predict_dense_default_batch_container.h */
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
//  Implementation of K-Nearest Neighbors algorithm container - a class that contains fast K-Nearest Neighbors prediction kernels for supported
//  architectures.
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Class containing computation methods for KD-tree based kNN model-based prediction
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    /**
     *  Computes the result of KD-tree based kNN model-based prediction
     */
    services::Status compute() override;
};

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictKernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    const classifier::prediction::Input * const input = static_cast<const classifier::prediction::Input *>(_in);
    Result * const result                             = static_cast<Result *>(_res);

    const data_management::NumericTableConstPtr a = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m             = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr r      = result->get(prediction::prediction);

    const Parameter * const par = static_cast<const Parameter *>(_par);

    data_management::NumericTablePtr indices;
    data_management::NumericTablePtr distances;
    if (par->resultsToCompute & computeIndicesOfNeighbors)
    {
        indices = result->get(prediction::indices);
    }
    if (par->resultsToCompute & computeDistances)
    {
        distances = result->get(prediction::distances);
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute, a.get(), m.get(),
                       r.get(), indices.get(), distances.get(), par);
}

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
