/* file: pca_dense_svd_batch_kernel_instance.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#include "src/algorithms/pca/pca_dense_svd_batch_kernel.h"
#include "src/algorithms/pca/pca_dense_svd_batch_impl.i"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
// Instantiated in pca_dense_base_fpt_cpu.cpp; without this MSVC warns C4661 here.
extern template class PCADenseBase<DAAL_FPTYPE, DAAL_CPU>;

template class DAAL_EXPORT PCASVDBatchKernel<DAAL_FPTYPE, interface3::BatchParameter<DAAL_FPTYPE, pca::svdDense>, DAAL_CPU>;

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
