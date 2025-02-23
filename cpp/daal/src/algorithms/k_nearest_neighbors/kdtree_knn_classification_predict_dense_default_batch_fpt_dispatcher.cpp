/* file: kdtree_knn_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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

#include "src/algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kdtree_knn_classification::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      kdtree_knn_classification::prediction::defaultDense)

namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface3
{
template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch()
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(const Batch<algorithmFPType, method> & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(size_t nClasses)
{
    parameter.nClasses = nClasses;
    initialize();
}

template class Batch<DAAL_FPTYPE, kdtree_knn_classification::prediction::defaultDense>;

} // namespace interface3
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
