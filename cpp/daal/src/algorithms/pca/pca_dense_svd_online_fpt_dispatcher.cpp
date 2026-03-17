/* file: pca_dense_svd_online_fpt_dispatcher.cpp */
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

//++
//  Implementation of PCA SVD algorithm container.
//--

#include "algorithms/pca/pca_online.h"
#include "src/algorithms/pca/pca_dense_svd_online_container.h"
#include "src/algorithms/pca/pca_dense_svd_online_kernel.h"
#include "src/algorithms/algorithm_dispatch_container_common.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(pca::internal::OnlineContainer, online, DAAL_FPTYPE, pca::svdDense)
namespace pca
{
namespace interface1
{
template <>
void Online<DAAL_FPTYPE, pca::svdDense>::initialize()
{
    _ac  = new __DAAL_ALGORITHM_CONTAINER(online, internal::OnlineContainer, DAAL_FPTYPE, svdDense)(&_env);
    _in  = &input;
    _par = &parameter;
    _partialResult.reset(new PartialResult<svdDense>());
    _result.reset(new ResultType());
}
template <>
DAAL_EXPORT Online<DAAL_FPTYPE, pca::svdDense>::Online()
{
    initialize();
}

using OnlineType = Online<DAAL_FPTYPE, pca::svdDense>;

template <>
DAAL_EXPORT OnlineType::Online(const OnlineType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
