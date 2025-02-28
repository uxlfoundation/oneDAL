/* file: pca_explained_variance_default_batch_fpt_dispatcher.cpp */
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
//  Instantiation of the container for single beta quality metrics.
//--
*/

#include "src/algorithms/pca/metrics/pca_explained_variance_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(pca::quality_metric::explained_variance::BatchContainer, batch, DAAL_FPTYPE,
                                      pca::quality_metric::explained_variance::defaultDense)
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch(size_t nFeatures, size_t nComponents) : parameter(nFeatures, nComponents)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : parameter(other.parameter)
{
    initialize();
    input.set(eigenvalues, other.input.get(eigenvalues));
}

} // namespace interface1
} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal
