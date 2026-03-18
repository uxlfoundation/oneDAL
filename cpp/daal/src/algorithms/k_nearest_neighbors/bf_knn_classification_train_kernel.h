/* file: bf_knn_classification_train_kernel.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_H__
#define __BF_KNN_CLASSIFICATION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "src/algorithms/kernel.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_training_types.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "src/algorithms/engines/engine.h"
namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, Model * r, const Parameter & par, engines::BatchBase & engine);
};

} // namespace internal
} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
