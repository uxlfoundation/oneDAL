/* file: em_gmm_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of EM calculation functions.
//--
*/

#include "src/algorithms/em/em_gmm_dense_default_batch_kernel.h"
#include "src/algorithms/em/em_gmm_dense_default_batch_impl.i"
#include "src/algorithms/em/em_gmm_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{

namespace internal
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

template class EMKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal
