/* file: kdtree_knn_classification_train_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of K-Nearest Neighbors training functions for the method of K-D Tree.
//--
*/

#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_train_container.h"
#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_train_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class DAAL_EXPORT KNNClassificationTrainBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
