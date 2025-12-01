/* file: spectral_embedding_default_dense_fpt_cpu.cpp */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Instantiation of CPU-specific spectral_embedding kernel implementations
//--
*/

#include "spectral_embedding_kernel.h"
#include "spectral_embedding_default_dense_impl.i"

namespace daal
{
namespace algorithms
{
namespace spectral_embedding
{
namespace internal
{
template class DAAL_EXPORT SpectralEmbeddingKernel<DAAL_FPTYPE, Method::defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace spectral_embedding
} // namespace algorithms
} // namespace daal
