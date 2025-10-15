/* file: low_order_moments_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of low order moments kernel.
//--
*/

#include "src/algorithms/low_order_moments/low_order_moments_batch_impl.i"
#include "src/algorithms/low_order_moments/low_order_moments_online_impl.i"
#include "src/algorithms/low_order_moments/low_order_moments_distributed_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
// fastCSR templates
template class DAAL_EXPORT LowOrderMomentsBatchKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
template class LowOrderMomentsOnlineKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
// singlePassCSR templates
template class LowOrderMomentsBatchKernel<DAAL_FPTYPE, singlePassCSR, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, singlePassCSR, DAAL_CPU>;
template class DAAL_EXPORT LowOrderMomentsOnlineKernel<DAAL_FPTYPE, singlePassCSR, DAAL_CPU>;
// sumCSR templates
template class LowOrderMomentsBatchKernel<DAAL_FPTYPE, sumCSR, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, sumCSR, DAAL_CPU>;
template class LowOrderMomentsOnlineKernel<DAAL_FPTYPE, sumCSR, DAAL_CPU>;
// singlePassDense templates
template class DAAL_EXPORT LowOrderMomentsBatchKernel<DAAL_FPTYPE, singlePassDense, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, singlePassDense, DAAL_CPU>;
template class DAAL_EXPORT LowOrderMomentsOnlineKernel<DAAL_FPTYPE, singlePassDense, DAAL_CPU>;
// sumDense templates
template class DAAL_EXPORT LowOrderMomentsBatchKernel<DAAL_FPTYPE, sumDense, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, sumDense, DAAL_CPU>;
template class DAAL_EXPORT LowOrderMomentsOnlineKernel<DAAL_FPTYPE, sumDense, DAAL_CPU>;
// defaultDense templates
template class DAAL_EXPORT LowOrderMomentsBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class LowOrderMomentsDistributedKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class DAAL_EXPORT LowOrderMomentsOnlineKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
