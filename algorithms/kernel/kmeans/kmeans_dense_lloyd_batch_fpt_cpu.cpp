/* file: kmeans_dense_lloyd_batch_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "kmeans_lloyd_kernel.h"
#include "kmeans_lloyd_batch_impl.i"
#include "kmeans_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, lloydDense, DAAL_CPU>;
}
namespace internal
{
template class KMeansBatchKernel<lloydDense, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
