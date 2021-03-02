/* file: kmeans_dense_lloyd_distr_step2_fpt_cpu_v1.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "src/algorithms/kmeans/inner/kmeans_container_v1.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
template class DistributedContainer<step2Master, DAAL_FPTYPE, lloydDense, DAAL_CPU>;
} // namespace interface1
} // namespace kmeans
} // namespace algorithms
} // namespace daal
