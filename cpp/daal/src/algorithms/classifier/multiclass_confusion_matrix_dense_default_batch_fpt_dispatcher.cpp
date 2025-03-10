/* file: multiclass_confusion_matrix_dense_default_batch_fpt_dispatcher.cpp */
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
//  Instantiation of the container for the multi-class confusion matrix.
//--
*/

#include "src/algorithms/classifier/multiclass_confusion_matrix_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(classifier::quality_metric::multiclass_confusion_matrix::BatchContainer, batch, DAAL_FPTYPE,
                                      classifier::quality_metric::multiclass_confusion_matrix::defaultDense)
namespace classifier
{
namespace quality_metric
{
namespace multiclass_confusion_matrix
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, classifier::quality_metric::multiclass_confusion_matrix::defaultDense>::Batch(size_t nClasses) : parameter(nClasses)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, classifier::quality_metric::multiclass_confusion_matrix::defaultDense>;
template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace multiclass_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal
