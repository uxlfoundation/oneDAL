/* file: pca_dense_correlation_batch_fpt_dispatcher.cpp */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#include "src/algorithms/pca/pca_dense_correlation_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(pca::interface3::BatchContainer, batch, DAAL_FPTYPE, pca::correlationDense)
namespace pca
{
namespace interface3
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, pca::correlationDense>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, pca::correlationDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal
