/* file: bf_knn_classification_predict_kernel.h */
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

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_H__
#define __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"

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
using namespace daal::data_management;
using namespace daal::internal;
using namespace algorithms::internal;

struct KernelParameter : bf_knn_classification::Parameter
{
    bf_knn_classification::VoteWeights voteWeights = bf_knn_classification::VoteWeights::voteUniform;
    PairwiseDistanceType pairwiseDistance          = PairwiseDistanceType::minkowski;
    double minkowskiDegree                         = 2.0;
};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationPredictKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * data, const classifier::Model * m, NumericTable * label, NumericTable * indices,
                             NumericTable * distances, const KernelParameter * par);
};

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
