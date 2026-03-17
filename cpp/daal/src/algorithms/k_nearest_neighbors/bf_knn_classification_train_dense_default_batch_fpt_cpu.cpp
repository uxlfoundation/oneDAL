/* file: bf_knn_classification_train_dense_default_batch_fpt_cpu.cpp */
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

#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_container.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel_impl.i"

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
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class DAAL_EXPORT KNNClassificationTrainKernel<DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
