/* file: bf_knn_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(bf_knn_classification::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      bf_knn_classification::prediction::defaultDense)

namespace bf_knn_classification
{
namespace prediction
{
namespace interface1
{

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch() : classifier::prediction::Batch()
{
    _par = new ParameterType();
    initialize();
}

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
}

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch(const Batch & other) : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
