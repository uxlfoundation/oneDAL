/* file: bf_knn_classification_predict_dense_default_batch_container.h */
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

#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h"
#include "services/error_indexes.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__PREDICTION__BATCHCONTAINER"></a>
 * \brief Class containing computation methods for BF kNN model-based prediction
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for BF kNN model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    /**
     *  Computes the result of BF kNN model-based prediction
     */
    services::Status compute() override;
};

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : PredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictKernel, algorithmFpType);
}

template <typename algorithmFpType, Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    const classifier::prediction::Input * const input        = static_cast<const classifier::prediction::Input *>(_in);
    bf_knn_classification::prediction::Result * const result = static_cast<bf_knn_classification::prediction::Result *>(_res);
    const data_management::NumericTableConstPtr a            = input->get(classifier::prediction::data);
    const classifier::ModelConstPtr m                        = input->get(classifier::prediction::model);
    const data_management::NumericTablePtr label             = result->get(bf_knn_classification::prediction::prediction);
    const data_management::NumericTablePtr indices           = result->get(bf_knn_classification::prediction::indices);
    const data_management::NumericTablePtr distances         = result->get(bf_knn_classification::prediction::distances);
    const Parameter * const par                              = static_cast<const Parameter *>(_par);

    internal::KernelParameter kernelPar;

    kernelPar.nClasses          = par->nClasses;
    kernelPar.k                 = par->k;
    kernelPar.dataUseInModel    = par->dataUseInModel;
    kernelPar.resultsToCompute  = par->resultsToCompute;
    kernelPar.voteWeights       = par->voteWeights;
    kernelPar.engine            = par->engine;
    kernelPar.resultsToEvaluate = par->resultsToEvaluate;

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationPredictKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType), compute, a.get(), m.get(),
                       label.get(), indices.get(), distances.get(), &kernelPar);
}

} // namespace internal

} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
