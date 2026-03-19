/* file: kmeans_csr_lloyd_distr_step1_fpt_cpu.cpp */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "src/algorithms/kmeans/kmeans_lloyd_kernel.h"
#include "src/algorithms/kmeans/kmeans_lloyd_distr_step1_impl.i"
#include "src/algorithms/kmeans/kmeans_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template class DistributedContainer<step1Local, DAAL_FPTYPE, kmeans::lloydCSR, DAAL_CPU>;
template class KMeansDistributedStep1Kernel<kmeans::lloydCSR, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
