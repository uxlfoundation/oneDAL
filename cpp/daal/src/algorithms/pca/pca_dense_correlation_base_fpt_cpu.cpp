/* file: pca_dense_correlation_base_fpt_cpu.cpp */
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

#include "src/algorithms/pca/pca_dense_correlation_base_impl.i"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template class DAAL_EXPORT PCACorrelationBase<DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
